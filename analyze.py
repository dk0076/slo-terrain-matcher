"""Grade analysis: turn the OSM network plus the DEM into matching segments.

The pipeline is:

1. Split every OSM way at its intersections into *edges*, so each edge is a
   stretch of road with nothing branching off it.
2. Chain edges into *corridors* (also called strokes): maximal runnable lines
   that continue straight through junctions. Without this step a two-mile
   climb would be invisible, because in town a single edge is rarely more than
   a couple of hundred metres long.
3. Resample each corridor at a fixed spacing, sample the DEM at every point,
   and smooth the elevation series before differencing it.
4. Slide a window over each profile and keep the windows whose length and
   mean grade fall inside a target's band.
5. Drop overlapping windows, then check whether the keepers can be looped.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import geo
from config import Config, Target

LOG = logging.getLogger("analyze")

#: ``surface`` tag values that mean a running-shoe-friendly hard surface.
PAVED_SURFACES = frozenset(
    {
        "asphalt",
        "paved",
        "concrete",
        "concrete:lanes",
        "concrete:plates",
        "chipseal",
        "paving_stones",
        "sett",
        "cobblestone",
        "unhewn_cobblestone",
        "bricks",
        "metal",
        "wood",
    }
)

#: ``surface`` tag values that mean dirt, gravel or worse.
UNPAVED_SURFACES = frozenset(
    {
        "unpaved",
        "gravel",
        "fine_gravel",
        "compacted",
        "dirt",
        "earth",
        "ground",
        "grass",
        "grass_paver",
        "sand",
        "mud",
        "pebblestone",
        "rock",
        "stone",
        "woodchips",
        "shells",
    }
)

#: Highway classes that are normally sealed even with no ``surface`` tag.
# Deliberately *not* used to claim a surface is paved; OSM surface tags are
# incomplete and absent means unknown. Kept only to explain the guess column.
USUALLY_PAVED_HIGHWAYS = frozenset(
    {"primary", "secondary", "tertiary", "residential", "living_street"}
)

#: A single-sample grade above this is almost certainly a DEM artefact.
DEM_SPIKE_GRADE_PCT = 25.0


# ---------------------------------------------------------------------------
# Network model
# ---------------------------------------------------------------------------


@dataclass
class Edge:
    """A stretch of road between two junctions (or a way's own endpoints)."""

    edge_id: int
    way_id: int
    name: str
    highway: str
    surface: str
    bridge: bool
    tunnel: bool
    start_node: int
    end_node: int
    coords: np.ndarray  # (n, 2) of (lon, lat)
    length_m: float
    # Tags the pleasantness score reads. Kept as the raw strings OSM uses, so
    # an unexpected value degrades to "unknown" rather than being guessed at.
    foot_tag: str = ""
    access_tag: str = ""
    sidewalk_tag: str = ""
    lit_tag: str = ""
    maxspeed_tag: str = ""
    lanes_tag: str = ""
    service_tag: str = ""

    @property
    def surface_class(self) -> str:
        """``"paved"``, ``"unpaved"`` or ``"unknown"`` for this edge."""
        value = self.surface.lower()
        if value in PAVED_SURFACES:
            return "paved"
        if value in UNPAVED_SURFACES:
            return "unpaved"
        return "unknown"

    def end_bearing(self, node: int, probe_m: float = 60.0) -> float:
        """Bearing of this edge as it *arrives at* ``node``.

        Args:
            node: Either ``start_node`` or ``end_node``.
            probe_m: How far back along the edge to measure from, so that a
                short wobble in the final vertex pair does not dominate.

        Returns:
            Bearing in degrees clockwise from north.

        Raises:
            ValueError: If ``node`` is not one of this edge's endpoints.
        """
        if node == self.end_node:
            coords = self.coords
        elif node == self.start_node:
            coords = self.coords[::-1]
        else:
            raise ValueError(f"node {node} is not an endpoint of edge {self.edge_id}")

        along = geo.cumulative_distance_m(coords)
        back_from = max(0.0, float(along[-1]) - probe_m)
        index = int(np.searchsorted(along, back_from, side="right")) - 1
        index = min(max(index, 0), len(coords) - 2)
        return geo.bearing_deg(
            coords[index][1], coords[index][0], coords[-1][1], coords[-1][0]
        )


@dataclass
class Corridor:
    """An ordered chain of edges forming one continuous runnable line."""

    corridor_id: int
    edges: list[Edge]
    coords: np.ndarray  # (n, 2) of (lon, lat), duplicates removed at joins
    node_sequence: list[int]  # junction node ids, len == len(edges) + 1
    node_offsets_m: np.ndarray  # along-corridor metres of each junction node

    @property
    def length_m(self) -> float:
        """Total length of the corridor in metres."""
        return float(self.node_offsets_m[-1])

    @property
    def names(self) -> list[str]:
        """Distinct street names along the corridor, in order of appearance."""
        seen: list[str] = []
        for edge in self.edges:
            if edge.name and edge.name not in seen:
                seen.append(edge.name)
        return seen

    def label(self) -> str:
        """A readable name for the corridor."""
        names = self.names
        if not names:
            classes = sorted({edge.highway for edge in self.edges})
            return f"unnamed {'/'.join(classes)}"
        if len(names) <= 2:
            return " / ".join(names)
        return f"{names[0]} to {names[-1]}"

    def edges_between(self, start_m: float, end_m: float) -> list[Edge]:
        """Edges overlapping the along-corridor interval ``[start_m, end_m]``."""
        result = []
        for index, edge in enumerate(self.edges):
            edge_start = float(self.node_offsets_m[index])
            edge_end = float(self.node_offsets_m[index + 1])
            if edge_end > start_m and edge_start < end_m:
                result.append(edge)
        return result

    def nodes_between(self, start_m: float, end_m: float) -> list[int]:
        """Junction nodes strictly inside the interval, plus the bounding two.

        Returns:
            The node ids whose offsets bracket ``[start_m, end_m]``, snapped
            outward to real junctions so a loop check has somewhere to start.
        """
        offsets = self.node_offsets_m
        first = int(np.searchsorted(offsets, start_m, side="right")) - 1
        last = int(np.searchsorted(offsets, end_m, side="left"))
        first = min(max(first, 0), len(offsets) - 1)
        last = min(max(last, 0), len(offsets) - 1)
        if last <= first:
            last = min(first + 1, len(offsets) - 1)
        return self.node_sequence[first : last + 1]


def build_edges(payload: dict[str, Any]) -> tuple[list[Edge], dict[int, tuple[float, float]]]:
    """Split OSM ways at their intersections into edges.

    A node is treated as a junction when it appears in more than one kept way,
    or more than once in the same way (a self-touching loop), or is a way's own
    first or last node.

    Args:
        payload: Overpass JSON with ``elements`` of type ``way``, each holding
            ``nodes`` and ``geometry``.

    Returns:
        A tuple of ``(edges, node_coords)`` where ``node_coords`` maps node id
        to ``(longitude, latitude)``.
    """
    ways = [
        element
        for element in payload.get("elements", [])
        if element.get("type") == "way"
        and element.get("geometry")
        and len(element.get("nodes") or ()) >= 2
    ]
    LOG.info("building edges from %d ways", len(ways))

    use_count: dict[int, int] = defaultdict(int)
    for way in ways:
        nodes = way["nodes"]
        for node in nodes:
            use_count[int(node)] += 1

    node_coords: dict[int, tuple[float, float]] = {}
    edges: list[Edge] = []
    next_id = 0

    for way in ways:
        tags = way.get("tags") or {}
        nodes = [int(node) for node in way["nodes"]]
        geometry = way["geometry"]
        if len(geometry) != len(nodes):
            # Overpass can clip geometry at the bbox edge; fall back to the
            # shorter of the two so indices stay aligned.
            limit = min(len(geometry), len(nodes))
            nodes, geometry = nodes[:limit], geometry[:limit]
            if len(nodes) < 2:
                continue

        coords = np.array(
            [[float(point["lon"]), float(point["lat"])] for point in geometry],
            dtype=float,
        )
        for node, point in zip(nodes, coords):
            node_coords.setdefault(node, (float(point[0]), float(point[1])))

        name = str(tags.get("name") or "")
        highway = str(tags.get("highway") or "")
        surface = str(tags.get("surface") or "")
        bridge = str(tags.get("bridge") or "no").lower() not in {"no", ""}
        tunnel = str(tags.get("tunnel") or "no").lower() not in {"no", ""}
        way_id = int(way.get("id", -1))
        extra = {
            "foot_tag": str(tags.get("foot") or "").lower(),
            "access_tag": str(tags.get("access") or "").lower(),
            "sidewalk_tag": str(tags.get("sidewalk") or "").lower(),
            "lit_tag": str(tags.get("lit") or "").lower(),
            "maxspeed_tag": str(tags.get("maxspeed") or ""),
            "lanes_tag": str(tags.get("lanes") or ""),
            "service_tag": str(tags.get("service") or "").lower(),
        }

        # Indices at which to cut the way.
        cuts = [0]
        for index in range(1, len(nodes) - 1):
            if use_count[nodes[index]] > 1:
                cuts.append(index)
        cuts.append(len(nodes) - 1)
        cuts = sorted(set(cuts))

        for left, right in zip(cuts[:-1], cuts[1:]):
            piece = coords[left : right + 1]
            if piece.shape[0] < 2:
                continue
            length = float(geo.cumulative_distance_m(piece)[-1])
            if length <= 0.0:
                continue
            edges.append(
                Edge(
                    edge_id=next_id,
                    way_id=way_id,
                    name=name,
                    highway=highway,
                    surface=surface,
                    bridge=bridge,
                    tunnel=tunnel,
                    start_node=nodes[left],
                    end_node=nodes[right],
                    coords=piece,
                    length_m=length,
                    **extra,
                )
            )
            next_id += 1

    total_km = sum(edge.length_m for edge in edges) / 1000.0
    LOG.info("  %d edges, %.0f km of network", len(edges), total_km)
    return edges, node_coords


def build_corridors(
    edges: Sequence[Edge],
    max_turn_deg: float = 60.0,
    same_name_bonus_deg: float = 20.0,
    max_corridor_m: float = 12000.0,
) -> list[Corridor]:
    """Chain edges into maximal straight-through corridors.

    Uses mutually-best pairing: at a junction, two edges are joined only when
    each is the other's straightest available continuation. That partitions
    the network into disjoint corridors without any arbitrary choices, and
    reproduces what a runner would call "the same road".

    Args:
        edges: Edges from :func:`build_edges`.
        max_turn_deg: Largest direction change allowed at a junction.
        same_name_bonus_deg: Extra allowance when the street name continues.
        max_corridor_m: Corridors are cut once they exceed this length.

    Returns:
        One :class:`Corridor` per chain. Every edge appears in exactly one.
    """
    by_node: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, edge in enumerate(edges):
        by_node[edge.start_node].append((index, edge.start_node))
        by_node[edge.end_node].append((index, edge.end_node))

    # best[(edge_index, node)] = the edge index it would rather continue into.
    best: dict[tuple[int, int], int] = {}
    for node, incident in by_node.items():
        if len(incident) < 2:
            continue
        for index, _ in incident:
            edge = edges[index]
            try:
                arrive = edge.end_bearing(node)
            except ValueError:  # pragma: no cover - defensive
                continue
            best_other: int | None = None
            best_turn = math.inf
            for other_index, _ in incident:
                if other_index == index:
                    continue
                other = edges[other_index]
                try:
                    other_arrive = other.end_bearing(node)
                except ValueError:  # pragma: no cover - defensive
                    continue
                # ``other_arrive`` points *into* the node; reversing it gives
                # the direction we would leave the node travelling along the
                # other edge. Continuing straight on means that departure
                # bearing matches the bearing we arrived on.
                depart = (other_arrive + 180.0) % 360.0
                turn = geo.turn_angle_deg(arrive, depart)
                allowance = max_turn_deg
                if edge.name and edge.name == other.name:
                    allowance += same_name_bonus_deg
                    turn -= same_name_bonus_deg / 2.0  # prefer name continuity
                if turn <= allowance and turn < best_turn:
                    best_turn = turn
                    best_other = other_index
            if best_other is not None:
                best[(index, node)] = best_other

    def partner(index: int, node: int) -> int | None:
        """The mutually-agreed continuation of ``index`` through ``node``."""
        other = best.get((index, node))
        if other is None:
            return None
        other_edge = edges[other]
        other_node = (
            other_edge.start_node
            if other_edge.start_node == node
            else other_edge.end_node
        )
        if best.get((other, other_node)) != index:
            return None
        return other

    assigned: set[int] = set()
    corridors: list[Corridor] = []

    for seed in range(len(edges)):
        if seed in assigned:
            continue
        chain = [seed]
        assigned.add(seed)

        # Walk forward from the seed's end node, then backward from its start.
        for direction in (1, -1):
            current = seed
            node = (
                edges[seed].end_node if direction == 1 else edges[seed].start_node
            )
            length = edges[seed].length_m
            while True:
                nxt = partner(current, node)
                if nxt is None or nxt in assigned:
                    break
                if length + edges[nxt].length_m > max_corridor_m:
                    break
                if direction == 1:
                    chain.append(nxt)
                else:
                    chain.insert(0, nxt)
                assigned.add(nxt)
                length += edges[nxt].length_m
                node = (
                    edges[nxt].end_node
                    if edges[nxt].start_node == node
                    else edges[nxt].start_node
                )
                current = nxt

        corridor = _assemble_corridor(len(corridors), [edges[i] for i in chain])
        if corridor is not None:
            corridors.append(corridor)

    lengths = np.array([corridor.length_m for corridor in corridors])
    LOG.info(
        "  %d corridors; median %.0f m, longest %.1f km",
        len(corridors),
        float(np.median(lengths)) if lengths.size else 0.0,
        float(lengths.max()) / 1000.0 if lengths.size else 0.0,
    )
    return corridors


def _assemble_corridor(corridor_id: int, chain: list[Edge]) -> Corridor | None:
    """Stitch an ordered edge chain into one polyline with node bookkeeping.

    Returns:
        The corridor, or ``None`` if the chain degenerates to nothing.
    """
    if not chain:
        return None

    # Orient the first edge so that the chain flows start -> end.
    if len(chain) > 1:
        second = chain[1]
        first_forward = chain[0].end_node in (second.start_node, second.end_node)
    else:
        first_forward = True

    pieces: list[np.ndarray] = []
    node_sequence: list[int] = []
    cursor: int | None = None

    for position, edge in enumerate(chain):
        if position == 0:
            forward = first_forward
        else:
            forward = edge.start_node == cursor
            if not forward and edge.end_node != cursor:
                # Chain broke (should not happen); stop here.
                break
        coords = edge.coords if forward else edge.coords[::-1]
        head = edge.start_node if forward else edge.end_node
        tail = edge.end_node if forward else edge.start_node

        if not pieces:
            pieces.append(coords)
            node_sequence.append(head)
        else:
            pieces.append(coords[1:])
        node_sequence.append(tail)
        cursor = tail

    used = chain[: len(node_sequence) - 1]
    if not used:
        return None

    coords = np.vstack(pieces)
    if coords.shape[0] < 2:
        return None

    offsets = [0.0]
    for edge in used:
        offsets.append(offsets[-1] + edge.length_m)

    return Corridor(
        corridor_id=corridor_id,
        edges=list(used),
        coords=coords,
        node_sequence=node_sequence,
        node_offsets_m=np.array(offsets, dtype=float),
    )


# ---------------------------------------------------------------------------
# Elevation sampling
# ---------------------------------------------------------------------------


class DemSampler:
    """Bilinear elevation lookup against a local DEM GeoTIFF.

    The DEM is read fully into memory once (a 0.2 degree clip of a 1/3
    arc-second DEM is roughly 18 MB), which makes the hundreds of thousands of
    point lookups the analysis needs cheap.

    Bilinear interpolation matters here: nearest-neighbour sampling against a
    10 m grid produces a staircase profile whose steps become fake pitches
    once differenced.
    """

    def __init__(self, dem_path: Path) -> None:
        """Open a DEM and prepare the coordinate transform.

        Args:
            dem_path: Path to a single-band elevation GeoTIFF.
        """
        import rasterio
        from pyproj import Transformer

        self.path = Path(dem_path)
        with rasterio.open(self.path) as src:
            band = src.read(1, masked=True)
            self.array = np.ma.filled(band.astype("float64"), np.nan)
            if src.nodata is not None:
                self.array[self.array == src.nodata] = np.nan
            # 3DEP uses a large negative sentinel in places.
            self.array[self.array < -1e4] = np.nan
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            self.res = src.res

        self._inverse = ~self.transform
        self._transformer = Transformer.from_crs(
            "EPSG:4326", self.crs, always_xy=True
        )
        self.rows, self.cols = self.array.shape
        LOG.info(
            "DEM loaded: %dx%d px, CRS %s, cell %.1f m",
            self.cols,
            self.rows,
            self.crs,
            self.nominal_cell_m,
        )

    @property
    def nominal_cell_m(self) -> float:
        """Approximate ground size of one DEM cell in metres."""
        mid_lat = (self.bounds.bottom + self.bounds.top) / 2.0
        return float(self.res[0]) * 111_320.0 * math.cos(math.radians(mid_lat))

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate elevation at WGS84 coordinates.

        Args:
            lon: Longitudes in degrees.
            lat: Latitudes in degrees.

        Returns:
            Elevations in metres, ``nan`` where the point falls outside the
            DEM or on a nodata cell.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x, y = self._transformer.transform(lon, lat)

        a, b, c, d, e, f = (
            self._inverse.a,
            self._inverse.b,
            self._inverse.c,
            self._inverse.d,
            self._inverse.e,
            self._inverse.f,
        )
        col_edge = a * x + b * y + c
        row_edge = d * x + e * y + f

        # Pixel centres sit half a cell in from the pixel corner.
        col = col_edge - 0.5
        row = row_edge - 0.5

        col0 = np.floor(col).astype(np.int64)
        row0 = np.floor(row).astype(np.int64)
        frac_col = col - col0
        frac_row = row - row0

        inside = (
            (col0 >= -1)
            & (col0 <= self.cols - 1)
            & (row0 >= -1)
            & (row0 <= self.rows - 1)
        )

        col0c = np.clip(col0, 0, self.cols - 1)
        col1c = np.clip(col0 + 1, 0, self.cols - 1)
        row0c = np.clip(row0, 0, self.rows - 1)
        row1c = np.clip(row0 + 1, 0, self.rows - 1)

        z00 = self.array[row0c, col0c]
        z01 = self.array[row0c, col1c]
        z10 = self.array[row1c, col0c]
        z11 = self.array[row1c, col1c]

        top = z00 * (1.0 - frac_col) + z01 * frac_col
        bottom = z10 * (1.0 - frac_col) + z11 * frac_col
        result = top * (1.0 - frac_row) + bottom * frac_row

        # If any corner was nodata the bilinear blend is nan; fall back to the
        # nearest valid corner so one bad cell does not void the sample.
        bad = np.isnan(result)
        if bad.any():
            stack = np.stack([z00, z01, z10, z11])
            with np.errstate(all="ignore"):
                fallback = np.nanmean(stack, axis=0)
            result = np.where(bad, fallback, result)

        return np.where(inside, result, np.nan)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that keeps the array length and the endpoints.

    Args:
        values: One-dimensional series, possibly containing ``nan``.
        window: Number of samples in the window. Values below 2 return the
            input unchanged. Even values are rounded up to the next odd number
            so the window stays centred.

    Returns:
        The smoothed series, same length as the input.
    """
    values = np.asarray(values, dtype=float)
    if window < 2 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window < 2:
        return values.copy()

    half = window // 2
    padded = np.pad(values, half, mode="edge")
    valid = (~np.isnan(padded)).astype(float)
    filled = np.where(np.isnan(padded), 0.0, padded)
    kernel = np.ones(window, dtype=float)
    totals = np.convolve(filled, kernel, mode="valid")
    counts = np.convolve(valid, kernel, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(counts > 0, totals / counts, np.nan)
    return result


@dataclass
class Profile:
    """A corridor's elevation/distance profile at uniform spacing."""

    corridor: Corridor
    points: np.ndarray  # (m, 2) of (lon, lat)
    dist_m: np.ndarray  # (m,)
    elev_raw_m: np.ndarray  # (m,)
    elev_m: np.ndarray  # (m,) smoothed
    spacing_m: float

    @property
    def size(self) -> int:
        """Number of samples in the profile."""
        return int(self.dist_m.size)

    def valid(self) -> bool:
        """Whether the profile has enough good samples to analyse."""
        return self.size >= 3 and bool(np.isfinite(self.elev_m).all())


def build_profiles(
    corridors: Sequence[Corridor],
    sampler: DemSampler,
    spacing_m: float,
    smoothing_samples: int,
    min_length_m: float,
) -> list[Profile]:
    """Sample the DEM along every corridor and smooth the result.

    Args:
        corridors: Corridors to profile.
        sampler: An open :class:`DemSampler`.
        spacing_m: Sample spacing along the corridor.
        smoothing_samples: Rolling-mean window applied before differencing.
        min_length_m: Corridors shorter than this are skipped entirely.

    Returns:
        Profiles for the corridors long enough to be worth analysing and that
        sit inside the DEM.
    """
    profiles: list[Profile] = []
    skipped_short = 0
    skipped_nodata = 0

    for index, corridor in enumerate(corridors, start=1):
        if corridor.length_m < min_length_m:
            skipped_short += 1
            continue

        points, dist = geo.resample_polyline(corridor.coords, spacing_m)
        if points.shape[0] < 3:
            skipped_short += 1
            continue

        raw = sampler.sample(points[:, 0], points[:, 1])
        finite = np.isfinite(raw)
        if finite.mean() < 0.9:
            skipped_nodata += 1
            continue
        if not finite.all():
            raw = np.interp(
                dist, dist[finite], raw[finite], left=np.nan, right=np.nan
            )
            if not np.isfinite(raw).all():
                skipped_nodata += 1
                continue

        smoothed = smooth_series(raw, smoothing_samples)
        profiles.append(
            Profile(
                corridor=corridor,
                points=points,
                dist_m=dist,
                elev_raw_m=raw,
                elev_m=smoothed,
                spacing_m=spacing_m,
            )
        )
        if index % 2000 == 0:
            LOG.info("  profiled %d/%d corridors", index, len(corridors))

    LOG.info(
        "  %d profiles built (%d too short, %d outside DEM)",
        len(profiles),
        skipped_short,
        skipped_nodata,
    )
    return profiles


# ---------------------------------------------------------------------------
# Window search
# ---------------------------------------------------------------------------


@dataclass
class Match:
    """One stretch of road that matches a target specification."""

    target_key: str
    corridor_id: int
    name: str
    start_index: int
    end_index: int
    start_m: float
    end_m: float
    length_m: float
    mean_grade_pct: float
    max_grade_pct: float
    min_grade_pct: float
    steepest_grade_pct: float
    gain_m: float
    loss_m: float
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    surface: str
    paved_fraction: float
    highways: str
    unreliable: bool
    unreliable_reasons: list[str] = field(default_factory=list)
    loopable: bool | None = None
    loop_length_m: float | None = None
    distance_from_reference_mi: float = 0.0
    score: float = 0.0
    coords: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    @property
    def length_mi(self) -> float:
        """Segment length in statute miles."""
        return geo.metres_to_miles(self.length_m)


def rolling_extreme_grades(
    profile: Profile, min_window_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grades measured over the shortest trustworthy sub-window.

    Point-to-point grade at 20 m spacing is dominated by DEM noise, so the
    "max grade" reported for a segment is the steepest grade sustained over
    ``min_window_m``, not the steepest single sample.

    Args:
        profile: The profile to measure.
        min_window_m: Length of the sub-window, metres.

    Returns:
        ``(start_distances, grades)`` where ``grades[i]`` is the grade from
        ``start_distances[i]`` over ``min_window_m``. Both are empty when the
        profile is shorter than the sub-window.
    """
    span = max(1, int(round(min_window_m / profile.spacing_m)))
    if profile.size <= span:
        return np.empty(0), np.empty(0)
    rise = profile.elev_m[span:] - profile.elev_m[:-span]
    run = profile.dist_m[span:] - profile.dist_m[:-span]
    with np.errstate(invalid="ignore", divide="ignore"):
        grades = np.where(run > 0, rise / run * 100.0, np.nan)
    return profile.dist_m[:-span], grades


def search_profile(
    profile: Profile,
    target: Target,
    min_window_m: float,
    step_samples: int,
    max_reversal_fraction: float = 0.25,
    max_undulation_m_per_km: float = 20.0,
) -> list[tuple[int, int, float]]:
    """Find every window in one profile matching a target's length and grade.

    Mean grade alone is not enough. A window that drops 40 m and climbs 20 m
    back has the same mean grade as a steady 20 m drop, but it is not the
    continuous descent the specification asks for. So a window is also
    required to be mostly monotonic: the elevation change running *against*
    the window's overall trend may not exceed ``max_reversal_fraction`` of the
    net change.

    For a target whose band straddles zero -- the flat finish -- "against the
    trend" has no meaning, so a different rule applies: total climbing per
    kilometre is capped. Otherwise a road of steep rollers that happens to
    end where it started would qualify as flat, which is not what anyone
    means by a flat finish.

    Args:
        profile: The profile to search.
        target: The specification to match.
        min_window_m: Hard floor on window length regardless of the target.
        step_samples: Stride between candidate window lengths and starts.
        max_reversal_fraction: Permitted backtracking, as a fraction of the
            window's net elevation change, for directional targets.
        max_undulation_m_per_km: Permitted total climbing per kilometre, for
            targets whose grade band straddles zero.

    Returns:
        A list of ``(start_index, end_index, mean_grade_pct)`` tuples, before
        any deduplication.
    """
    spacing = profile.spacing_m
    lo_m = max(target.length_m[0], min_window_m)
    hi_m = target.length_m[1]
    if hi_m < lo_m or profile.dist_m[-1] < lo_m:
        return []

    span_lo = max(1, int(math.ceil(lo_m / spacing)))
    span_hi = min(profile.size - 1, int(math.floor(hi_m / spacing)))
    if span_hi < span_lo:
        return []

    step = max(1, step_samples)
    grade_lo, grade_hi = target.grade_pct
    results: list[tuple[int, int, float]] = []
    elev = profile.elev_m
    dist = profile.dist_m

    # A band spanning zero (the flat target) has no meaningful trend, so it
    # gets the undulation cap instead of the monotonicity rule.
    check_continuity = not (grade_lo < 0.0 < grade_hi)
    diffs = np.diff(elev)
    gain_prefix = np.concatenate([[0.0], np.cumsum(np.clip(diffs, 0, None))])
    loss_prefix = np.concatenate([[0.0], np.cumsum(-np.clip(diffs, None, 0))])

    for span in range(span_lo, span_hi + 1, step):
        rise = elev[span:] - elev[:-span]
        run = dist[span:] - dist[:-span]
        with np.errstate(invalid="ignore", divide="ignore"):
            grades = np.where(run > 0, rise / run * 100.0, np.nan)
        keep = (grades >= grade_lo) & (grades <= grade_hi) & (run >= lo_m)

        gain = gain_prefix[span:] - gain_prefix[:-span]
        if check_continuity:
            loss = loss_prefix[span:] - loss_prefix[:-span]
            against = np.where(rise >= 0.0, loss, gain)
            allowance = max_reversal_fraction * np.abs(rise)
            keep &= against <= allowance
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                per_km = np.where(run > 0, gain / (run / 1000.0), np.inf)
            keep &= per_km <= max_undulation_m_per_km

        hits = np.flatnonzero(keep)
        if hits.size == 0:
            continue
        if step > 1:
            hits = hits[hits % step == 0]
        for start in hits:
            results.append((int(start), int(start) + span, float(grades[start])))

    return results


def dedupe_windows(
    candidates: Iterable[tuple[int, int, float]], target: Target
) -> list[tuple[int, int, float]]:
    """Collapse overlapping windows from the same profile.

    Preference order is closeness to the target's ideal grade first, then
    length, so an overlapping cluster collapses to the window that best
    reproduces the race feature rather than whichever one came first.

    Args:
        candidates: ``(start, end, grade)`` tuples from one profile.
        target: The target being matched, for the ideal grade.

    Returns:
        A non-overlapping subset, ordered along the profile.
    """
    ranked = sorted(
        candidates,
        key=lambda item: (
            abs(item[2] - target.ideal_grade_pct),
            -(item[1] - item[0]),
        ),
    )
    kept: list[tuple[int, int, float]] = []
    for start, end, grade in ranked:
        if any(start < other_end and end > other_start for other_start, other_end, _ in kept):
            continue
        kept.append((start, end, grade))
    return sorted(kept, key=lambda item: item[0])


def _window_surface(
    edges: Sequence[Edge], corridor: Corridor, start_m: float, end_m: float
) -> tuple[str, float]:
    """Length-weighted surface classification for a window.

    Returns:
        ``(classification, paved_fraction)`` where classification is
        ``"paved"``, ``"unpaved"`` or ``"unknown"``. An absent OSM ``surface``
        tag counts as unknown, never as paved.
    """
    totals = {"paved": 0.0, "unpaved": 0.0, "unknown": 0.0}
    for index, edge in enumerate(corridor.edges):
        edge_start = float(corridor.node_offsets_m[index])
        edge_end = float(corridor.node_offsets_m[index + 1])
        overlap = min(edge_end, end_m) - max(edge_start, start_m)
        if overlap <= 0.0:
            continue
        totals[edge.surface_class] += overlap

    total = sum(totals.values())
    if total <= 0.0:
        return "unknown", 0.0
    paved_fraction = totals["paved"] / total
    if totals["unpaved"] / total > 0.1:
        return "unpaved", paved_fraction
    if paved_fraction >= 0.9:
        return "paved", paved_fraction
    return "unknown", paved_fraction


def _window_reliability(
    profile: Profile,
    edges: Sequence[Edge],
    start_index: int,
    end_index: int,
    waterway_edges: frozenset[int],
) -> tuple[bool, list[str]]:
    """Decide whether the DEM can be trusted over this window.

    The DEM is bare earth, so a bridge or overpass reads as the ground
    underneath it and a culvert-free creek crossing can read as a notch.

    Returns:
        ``(unreliable, reasons)``.
    """
    reasons: list[str] = []
    if any(edge.bridge for edge in edges):
        reasons.append("bridge")
    if any(edge.tunnel for edge in edges):
        reasons.append("tunnel")
    if any(edge.edge_id in waterway_edges for edge in edges):
        reasons.append("crosses waterway")

    raw = profile.elev_raw_m[start_index : end_index + 1]
    dist = profile.dist_m[start_index : end_index + 1]
    if raw.size >= 2:
        run = np.diff(dist)
        with np.errstate(invalid="ignore", divide="ignore"):
            point_grades = np.where(run > 0, np.diff(raw) / run * 100.0, 0.0)
        if np.nanmax(np.abs(point_grades)) > DEM_SPIKE_GRADE_PCT:
            reasons.append("DEM spike")

    return bool(reasons), reasons


def materialise_match(
    profile: Profile,
    target: Target,
    start_index: int,
    end_index: int,
    mean_grade: float,
    extremes: tuple[np.ndarray, np.ndarray],
    min_window_m: float,
    waterway_edges: frozenset[int],
    reference_point: tuple[float, float],
) -> Match:
    """Turn a matching index window into a fully described :class:`Match`."""
    corridor = profile.corridor
    dist = profile.dist_m
    start_m = float(dist[start_index])
    end_m = float(dist[end_index])
    length_m = end_m - start_m

    elev = profile.elev_m[start_index : end_index + 1]
    diffs = np.diff(elev)
    gain = float(diffs[diffs > 0].sum()) if diffs.size else 0.0
    loss = float(-diffs[diffs < 0].sum()) if diffs.size else 0.0

    extreme_starts, extreme_grades = extremes
    if extreme_starts.size:
        inside = (extreme_starts >= start_m) & (
            extreme_starts <= max(start_m, end_m - min_window_m)
        )
        window_grades = extreme_grades[inside]
    else:
        window_grades = np.empty(0)
    if window_grades.size and np.isfinite(window_grades).any():
        max_grade = float(np.nanmax(window_grades))
        min_grade = float(np.nanmin(window_grades))
    else:
        max_grade = min_grade = mean_grade
    # "Max grade" means the steepest grade sustained in the direction the
    # segment actually runs: for a descent that is the most negative value,
    # not the most positive one.
    steepest = min_grade if mean_grade < 0 else max_grade

    points = profile.points[start_index : end_index + 1]
    edges = corridor.edges_between(start_m, end_m)
    surface, paved_fraction = _window_surface(edges, corridor, start_m, end_m)
    unreliable, reasons = _window_reliability(
        profile, edges, start_index, end_index, waterway_edges
    )

    names: list[str] = []
    for edge in edges:
        if edge.name and edge.name not in names:
            names.append(edge.name)
    if not names:
        names = [corridor.label()]
    highways = "/".join(sorted({edge.highway for edge in edges if edge.highway}))

    ref_lat, ref_lon = reference_point
    distance_mi = geo.metres_to_miles(
        geo.haversine_m(
            ref_lat, ref_lon, float(points[0][1]), float(points[0][0])
        )
    )

    return Match(
        target_key=target.key,
        corridor_id=corridor.corridor_id,
        name=" / ".join(names[:3]),
        start_index=start_index,
        end_index=end_index,
        start_m=start_m,
        end_m=end_m,
        length_m=length_m,
        mean_grade_pct=mean_grade,
        max_grade_pct=max_grade,
        min_grade_pct=min_grade,
        steepest_grade_pct=steepest,
        gain_m=gain,
        loss_m=loss,
        start_lat=float(points[0][1]),
        start_lon=float(points[0][0]),
        end_lat=float(points[-1][1]),
        end_lon=float(points[-1][0]),
        surface=surface,
        paved_fraction=paved_fraction,
        highways=highways,
        unreliable=unreliable,
        unreliable_reasons=reasons,
        distance_from_reference_mi=distance_mi,
        coords=points.copy(),
    )


def find_matches(
    profiles: Sequence[Profile],
    config: Config,
    waterway_edges: frozenset[int] = frozenset(),
) -> dict[str, list[Match]]:
    """Search every profile for every configured target.

    Args:
        profiles: Profiles from :func:`build_profiles`.
        config: Loaded configuration.
        waterway_edges: Edge ids known to cross a waterway.

    Returns:
        Matches keyed by target key, each list sorted by descending score.
    """
    analysis = config.analysis
    min_window_m = float(analysis.get("min_window_m", 100.0))
    step = int(analysis.get("window_step_samples", 2))
    reversal = float(analysis.get("max_reversal_fraction", 0.25))
    undulation = float(analysis.get("max_undulation_m_per_km", 20.0))
    reference = config.reference_point

    results: dict[str, list[Match]] = {target.key: [] for target in config.targets}
    extremes_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for count, profile in enumerate(profiles, start=1):
        extremes = extremes_cache.get(profile.corridor.corridor_id)
        if extremes is None:
            extremes = rolling_extreme_grades(profile, min_window_m)
            extremes_cache[profile.corridor.corridor_id] = extremes

        for target in config.targets:
            raw = search_profile(
                profile, target, min_window_m, step, reversal, undulation
            )
            if not raw:
                continue
            for start, end, grade in dedupe_windows(raw, target):
                match = materialise_match(
                    profile,
                    target,
                    start,
                    end,
                    grade,
                    extremes,
                    min_window_m,
                    waterway_edges,
                    reference,
                )
                if target.require_paved and match.surface != "paved":
                    continue
                results[target.key].append(match)

        if count % 2000 == 0:
            LOG.info("  searched %d/%d profiles", count, len(profiles))

    for target in config.targets:
        matches = results[target.key]
        for match in matches:
            match.score = score_match(match, target, config)
        matches.sort(key=lambda item: item.score, reverse=True)
        LOG.info("  %-16s %d candidate segments", target.key, len(matches))

    return results


# ---------------------------------------------------------------------------
# Scoring and loopability
# ---------------------------------------------------------------------------


def score_match(match: Match, target: Target, config: Config) -> float:
    """Rate how well a segment reproduces its target, from 0 to about 100.

    The score combines closeness to the ideal grade, how well the length fits
    the target band, a bonus for a known-paved surface, and a bonus for being
    close to the reference point. Segments the DEM cannot be trusted on are
    penalised rather than dropped, so they still appear but rank low.

    Args:
        match: The candidate segment.
        target: The target it was matched against.
        config: Loaded configuration, for the scoring weights.

    Returns:
        The quality score.
    """
    scoring = config.scoring
    grade_weight = float(scoring.get("grade_weight", 45.0))
    length_weight = float(scoring.get("length_weight", 25.0))
    paved_bonus = float(scoring.get("paved_bonus", 15.0))
    proximity_bonus = float(scoring.get("proximity_bonus", 15.0))
    decay_mi = max(0.1, float(scoring.get("proximity_decay_mi", 8.0)))
    penalty = float(scoring.get("unreliable_penalty", 25.0))

    # Grade: full marks at the ideal, zero at the edge of the band.
    lo, hi = target.grade_pct
    half_band = max(1e-6, (hi - lo) / 2.0)
    grade_error = abs(match.mean_grade_pct - target.ideal_grade_pct)
    reach = max(
        abs(target.ideal_grade_pct - lo), abs(hi - target.ideal_grade_pct), half_band
    )
    grade_score = grade_weight * max(0.0, 1.0 - grade_error / reach)

    # Length: full marks at or above the target's upper bound for open-ended
    # features, otherwise at the middle of the band.
    length_lo, length_hi = target.length_m
    if match.length_m >= length_hi:
        length_fit = 1.0
    elif match.length_m <= length_lo:
        length_fit = 0.0
    else:
        length_fit = (match.length_m - length_lo) / (length_hi - length_lo)
    length_score = length_weight * length_fit

    # Surface: paved earns the bonus, a known-unpaved surface is penalised,
    # and an absent tag sits in between -- absent means unknown, not paved,
    # so it neither earns the full bonus nor takes the penalty.
    unpaved_penalty = float(scoring.get("unpaved_penalty", 20.0))
    surface_score = 0.0
    if target.prefer_paved:
        if match.surface == "paved":
            surface_score = paved_bonus
        elif match.surface == "unpaved":
            surface_score = -unpaved_penalty
        else:
            surface_score = paved_bonus * 0.4 * match.paved_fraction
    else:
        surface_score = paved_bonus * 0.5

    proximity = max(
        0.0, 1.0 - match.distance_from_reference_mi / decay_mi
    )
    proximity_score = proximity_bonus * proximity

    total = grade_score + length_score + surface_score + proximity_score
    if match.unreliable:
        total -= penalty
    if target.want_loopable and match.loopable:
        total += 5.0
    return round(max(0.0, total), 2)


def annotate_loopability(
    matches: dict[str, list[Match]],
    corridors: Sequence[Corridor],
    edges: Sequence[Edge],
    config: Config,
) -> None:
    """Fill in ``loopable`` for the top candidates of loop-seeking targets.

    A segment is loopable when a return route exists between its endpoints
    that does not reuse the segment itself and is not much longer than it, so
    the climb can be repeated without an out-and-back on the same road.

    This runs a shortest-path query per candidate, so it is limited to the top
    ``analysis.loop_check_limit`` candidates per target. Matches are re-scored
    afterwards, since loopability feeds the score.

    Args:
        matches: Matches keyed by target key, highest scoring first.
        corridors: All corridors, indexed by ``corridor_id``.
        edges: All edges, used to build the routing graph.
        config: Loaded configuration.
    """
    import networkx as nx

    targets = {target.key: target for target in config.targets}
    wanted = [key for key, target in targets.items() if target.want_loopable]
    if not wanted:
        return

    limit = int(config.analysis.get("loop_check_limit", 60))
    max_ratio = float(config.analysis.get("loop_max_ratio", 3.0))

    LOG.info("checking loopability for %d target(s)", len(wanted))
    graph = nx.MultiGraph()
    for edge in edges:
        graph.add_edge(
            edge.start_node,
            edge.end_node,
            key=edge.edge_id,
            weight=edge.length_m,
        )

    by_id = {corridor.corridor_id: corridor for corridor in corridors}

    for key in wanted:
        checked = 0
        for match in matches.get(key, []):
            if checked >= limit:
                break
            corridor = by_id.get(match.corridor_id)
            if corridor is None:
                continue
            nodes = corridor.nodes_between(match.start_m, match.end_m)
            if len(nodes) < 2:
                match.loopable = False
                checked += 1
                continue

            start_node, end_node = nodes[0], nodes[-1]
            inside = {
                edge.edge_id for edge in corridor.edges_between(match.start_m, match.end_m)
            }
            removed = [
                (u, v, k, data)
                for u, v, k, data in graph.edges(keys=True, data=True)
                if k in inside
            ]
            graph.remove_edges_from([(u, v, k) for u, v, k, _ in removed])
            try:
                return_length = nx.shortest_path_length(
                    graph, start_node, end_node, weight="weight"
                )
                match.loopable = return_length <= max_ratio * match.length_m
                match.loop_length_m = float(return_length) + match.length_m
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                match.loopable = False
                match.loop_length_m = None
            finally:
                for u, v, k, data in removed:
                    graph.add_edge(u, v, key=k, **data)
            checked += 1

        target = targets[key]
        for match in matches.get(key, []):
            match.score = score_match(match, target, config)
        matches[key].sort(key=lambda item: item.score, reverse=True)
        loopable_count = sum(
            1 for match in matches.get(key, []) if match.loopable
        )
        LOG.info("  %-16s %d of %d checked are loopable", key, loopable_count, checked)


# ---------------------------------------------------------------------------
# Waterway crossings
# ---------------------------------------------------------------------------


def find_waterway_crossings(
    edges: Sequence[Edge], waterways: dict[str, Any] | None
) -> frozenset[int]:
    """Identify edges that cross a mapped waterway.

    Args:
        edges: Road edges to test.
        waterways: Overpass payload of waterway ways, or ``None`` to skip.

    Returns:
        The set of edge ids that intersect a waterway centreline.
    """
    if not waterways:
        return frozenset()

    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    lines = []
    for element in waterways.get("elements", []):
        geometry = element.get("geometry") or []
        coords = [
            (float(point["lon"]), float(point["lat"]))
            for point in geometry
            if point.get("lon") is not None
        ]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    if not lines:
        return frozenset()

    tree = STRtree(lines)
    crossing: set[int] = set()
    for edge in edges:
        if edge.coords.shape[0] < 2:
            continue
        road = LineString(edge.coords)
        for index in tree.query(road):
            if road.intersects(lines[int(index)]):
                crossing.add(edge.edge_id)
                break

    LOG.info(
        "  %d of %d edges cross a mapped waterway", len(crossing), len(edges)
    )
    return frozenset(crossing)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Everything the reporting step needs."""

    config: Config
    edges: list[Edge]
    corridors: list[Corridor]
    profiles: list[Profile]
    matches: dict[str, list[Match]]
    dem_cell_m: float


def run_analysis(
    config: Config,
    dem_path: Path,
    osm_payload: dict[str, Any],
    waterways: dict[str, Any] | None = None,
) -> AnalysisResult:
    """Run the whole grade analysis end to end.

    Args:
        config: Loaded configuration.
        dem_path: Path to the clipped DEM GeoTIFF.
        osm_payload: Overpass JSON of the road network.
        waterways: Optional Overpass JSON of waterways, for crossing flags.

    Returns:
        The analysis result, ready for :mod:`report`.
    """
    analysis = config.analysis
    edges, _ = build_edges(osm_payload)
    corridors = build_corridors(
        edges,
        max_turn_deg=float(analysis.get("max_turn_angle_deg", 60.0)),
        same_name_bonus_deg=float(analysis.get("same_name_turn_bonus_deg", 20.0)),
        max_corridor_m=float(analysis.get("max_corridor_m", 12000.0)),
    )

    waterway_edges = find_waterway_crossings(edges, waterways)

    sampler = DemSampler(dem_path)
    shortest_target_m = min(
        max(target.length_m[0], float(analysis.get("min_window_m", 100.0)))
        for target in config.targets
    )
    profiles = build_profiles(
        corridors,
        sampler,
        spacing_m=float(analysis.get("sample_spacing_m", 20.0)),
        smoothing_samples=int(analysis.get("smoothing_samples", 5)),
        min_length_m=shortest_target_m,
    )

    matches = find_matches(profiles, config, waterway_edges)
    annotate_loopability(matches, corridors, edges, config)

    return AnalysisResult(
        config=config,
        edges=edges,
        corridors=corridors,
        profiles=profiles,
        matches=matches,
        dem_cell_m=sampler.nominal_cell_m,
    )
