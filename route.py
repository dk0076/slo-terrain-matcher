"""Build whole routes that reproduce the race course's gradient profile.

The segment search in :mod:`analyze` answers "where can I find a 5% pitch?".
This module answers a different question: "what single route should I run?"

It does that with a beam search over the road graph. The course is reduced to
a curve of grade against *fractional* distance -- what grade the course is
doing 30% of the way in, 70% of the way in, and so on. A candidate route is
then grown edge by edge, and at each step the edges that keep the route's
grade close to the course's grade *at the same fraction of the way through*
are preferred.

Working in fractions rather than absolute distance is what lets one mechanism
produce both outputs. For a full-distance simulation the fraction maps
one-to-one onto race miles. For a shorter signature loop the same curve is
compressed, so the loop meets the course's features in the same order and
proportion, at their true grades, over a distance you can actually run on a
weeknight.
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

import geo
from analyze import Edge, smooth_series
from config import Config

LOG = logging.getLogger("route")

#: Metres to feet.
FEET_PER_METRE = 3.280839895


# ---------------------------------------------------------------------------
# The target: the course as grade against fractional distance
# ---------------------------------------------------------------------------


@dataclass
class CourseTarget:
    """The race course expressed as grade against fraction of the way through.

    It also carries the course's own elevation curve. Matching grade alone is
    not enough: a route can hold the right grade at every point and still end
    up far below where the course would be, because small errors accumulate.
    The elevation curve is what lets the search notice that and correct.
    """

    fractions: np.ndarray  # (n,) in [0, 1]
    grades_pct: np.ndarray  # (n,)
    total_m: float
    total_gain_m: float
    window_m: float
    elev_fractions: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    elev_delta_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    def grade_at(self, fraction: float | np.ndarray) -> np.ndarray:
        """The course's grade at a fractional position along it."""
        return np.interp(
            np.clip(fraction, 0.0, 1.0), self.fractions, self.grades_pct
        )

    def elevation_delta_at(
        self, fraction: float | np.ndarray, length_scale: float = 1.0
    ) -> np.ndarray:
        """How far above its start the course is, at a fractional position.

        Args:
            fraction: Position along the course, 0 to 1.
            length_scale: Ratio of the replica's length to the course's.
                Elevation scales with it, which is what keeps the grades the
                same when the course is compressed into a shorter route.

        Returns:
            Metres above the start.
        """
        return (
            np.interp(
                np.clip(fraction, 0.0, 1.0), self.elev_fractions, self.elev_delta_m
            )
            * length_scale
        )

    def gain_for_length(self, length_m: float) -> float:
        """Climbing a route of ``length_m`` would have at the course's rate.

        A shorter route reproducing the same *grades* in the same order
        accumulates climbing in proportion to its length.
        """
        return self.total_gain_m * (length_m / self.total_m)


def course_target(
    course_profile, window_m: float = 400.0
) -> CourseTarget:
    """Reduce a course profile to a grade-against-fraction curve.

    Args:
        course_profile: A :class:`course.CourseProfile`.
        window_m: Window over which course grade is measured. Roughly a
            quarter mile, which is the scale a runner actually feels and is
            long enough that GPS elevation noise does not dominate.

    Returns:
        The target curve.
    """
    dist = course_profile.dist_m
    elev = course_profile.elev_m
    total = float(dist[-1])

    span = max(1, int(round(window_m / course_profile.spacing_m)))
    if dist.size <= span:
        raise ValueError("course is shorter than one grade window")

    rise = elev[span:] - elev[:-span]
    run = dist[span:] - dist[:-span]
    with np.errstate(invalid="ignore", divide="ignore"):
        grades = np.where(run > 0, rise / run * 100.0, 0.0)
    centres = (dist[span:] + dist[:-span]) / 2.0

    diffs = np.diff(elev)
    gain = float(diffs[diffs > 0].sum())

    # The course's own elevation curve, relative to its start.
    elev_fractions = dist / total
    elev_delta = elev - elev[0]

    LOG.info(
        "course target: %.2f mi, %.0f ft gain, grade %.2f%% to %+.2f%%",
        geo.metres_to_miles(total),
        gain * FEET_PER_METRE,
        float(grades.min()),
        float(grades.max()),
    )
    return CourseTarget(
        fractions=centres / total,
        grades_pct=grades,
        total_m=total,
        total_gain_m=gain,
        window_m=window_m,
        elev_fractions=elev_fractions,
        elev_delta_m=elev_delta,
    )


# ---------------------------------------------------------------------------
# The road graph, with elevation on every edge
# ---------------------------------------------------------------------------


@dataclass
class RoutableEdge:
    """An edge with its sampled elevation, ready for routing."""

    edge: Edge
    points: np.ndarray  # (n, 2) of (lon, lat), start -> end
    elevations: np.ndarray  # (n,)

    @property
    def rise_m(self) -> float:
        """Net elevation change travelling start to end."""
        return float(self.elevations[-1] - self.elevations[0])

    @property
    def length_m(self) -> float:
        """Edge length in metres."""
        return self.edge.length_m

    def grade_pct(self, forward: bool) -> float:
        """Mean grade travelling in the given direction, in percent."""
        if self.length_m <= 0:
            return 0.0
        rise = self.rise_m if forward else -self.rise_m
        return rise / self.length_m * 100.0

    def traverse(self, forward: bool) -> tuple[np.ndarray, np.ndarray]:
        """Points and elevations in travel order."""
        if forward:
            return self.points, self.elevations
        return self.points[::-1], self.elevations[::-1]


class RoadGraph:
    """The road network with elevation, indexed for routing."""

    def __init__(
        self,
        edges: Sequence[Edge],
        sampler,
        spacing_m: float = 20.0,
        smoothing_samples: int = 5,
    ) -> None:
        """Sample the DEM along every edge and build the adjacency index.

        All edges are sampled in one vectorised DEM lookup, which is far
        faster than sampling them one at a time.

        Args:
            edges: Edges from :func:`analyze.build_edges`.
            sampler: An open :class:`analyze.DemSampler`.
            spacing_m: Sample spacing along each edge.
            smoothing_samples: Rolling-mean window for elevation.
        """
        LOG.info("sampling elevation along %d edges", len(edges))

        chunks: list[np.ndarray] = []
        bounds: list[tuple[int, int]] = []
        kept: list[Edge] = []
        cursor = 0
        for edge in edges:
            points, _ = geo.resample_polyline(edge.coords, spacing_m)
            if points.shape[0] < 2:
                continue
            chunks.append(points)
            bounds.append((cursor, cursor + points.shape[0]))
            cursor += points.shape[0]
            kept.append(edge)

        if not chunks:
            raise ValueError("no edges could be sampled")

        every_point = np.vstack(chunks)
        every_elevation = sampler.sample(every_point[:, 0], every_point[:, 1])
        LOG.info("  %d elevation samples", every_point.shape[0])

        self.edges: dict[int, RoutableEdge] = {}
        self.adjacency: dict[int, list[tuple[int, int]]] = {}
        dropped = 0

        for edge, (start, end) in zip(kept, bounds):
            points = every_point[start:end]
            elevations = every_elevation[start:end]
            finite = np.isfinite(elevations)
            if finite.mean() < 0.9:
                dropped += 1
                continue
            if not finite.all():
                elevations = np.interp(
                    np.arange(elevations.size),
                    np.flatnonzero(finite),
                    elevations[finite],
                )
            if elevations.size >= smoothing_samples:
                elevations = smooth_series(elevations, smoothing_samples)

            routable = RoutableEdge(
                edge=edge, points=points, elevations=elevations
            )
            self.edges[edge.edge_id] = routable
            self.adjacency.setdefault(edge.start_node, []).append(
                (edge.edge_id, edge.end_node)
            )
            self.adjacency.setdefault(edge.end_node, []).append(
                (edge.edge_id, edge.start_node)
            )

        LOG.info(
            "  routable graph: %d edges, %d nodes (%d dropped outside DEM)",
            len(self.edges),
            len(self.adjacency),
            dropped,
        )

    def node_coord(self, node: int) -> tuple[float, float]:
        """A representative ``(lon, lat)`` for a node."""
        for edge_id, _ in self.adjacency.get(node, []):
            routable = self.edges[edge_id]
            if routable.edge.start_node == node:
                return tuple(routable.points[0])
            return tuple(routable.points[-1])
        raise KeyError(f"node {node} is not in the graph")

    def nearest_node(self, lat: float, lon: float) -> int:
        """The graph node closest to a coordinate.

        Raises:
            ValueError: If the graph has no nodes.
        """
        best_node = None
        best_distance = math.inf
        for node in self.adjacency:
            try:
                node_lon, node_lat = self.node_coord(node)
            except KeyError:  # pragma: no cover - defensive
                continue
            distance = (node_lat - lat) ** 2 + (node_lon - lon) ** 2
            if distance < best_distance:
                best_distance = distance
                best_node = node
        if best_node is None:
            raise ValueError("graph has no nodes")
        return best_node

    def distances_from(self, source: int) -> dict[int, float]:
        """Shortest-path distance from ``source`` to every reachable node.

        Used both to prune routes that have wandered too far to get home and
        to close a loop at the end.
        """
        distances = {source: 0.0}
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances.get(node, math.inf):
                continue
            for edge_id, neighbour in self.adjacency.get(node, ()):
                candidate = distance + self.edges[edge_id].length_m
                if candidate < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
        return distances

    def shortest_path_edges(self, source: int, destination: int) -> list[int]:
        """Edge ids along the shortest path, or ``[]`` if unreachable."""
        if source == destination:
            return []
        previous: dict[int, tuple[int, int]] = {}
        distances = {source: 0.0}
        queue: list[tuple[float, int]] = [(0.0, source)]
        seen: set[int] = set()
        while queue:
            distance, node = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            if node == destination:
                break
            for edge_id, neighbour in self.adjacency.get(node, ()):
                candidate = distance + self.edges[edge_id].length_m
                if candidate < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    previous[neighbour] = (node, edge_id)
                    heapq.heappush(queue, (candidate, neighbour))

        if destination not in previous and destination != source:
            return []
        path: list[int] = []
        cursor = destination
        while cursor != source:
            cursor, edge_id = previous[cursor]
            path.append(edge_id)
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@dataclass
class Route:
    """A built route with its profile and how well it matches the course."""

    name: str
    edge_ids: list[int]
    coords: np.ndarray  # (n, 2) of (lon, lat)
    elevations: np.ndarray  # (n,)
    dist_m: np.ndarray  # (n,)
    names: list[str]
    surface_fractions: dict[str, float]
    closed_loop: bool
    target: CourseTarget
    fit: dict[str, float] = field(default_factory=dict)
    _edge_lengths: dict[int, float] = field(default_factory=dict)

    @property
    def length_mi(self) -> float:
        """Route length in miles."""
        return geo.metres_to_miles(float(self.dist_m[-1]))

    @property
    def gain_m(self) -> float:
        """Total climbing in metres."""
        diffs = np.diff(self.elevations)
        return float(diffs[diffs > 0].sum())

    @property
    def repeat_fraction(self) -> float:
        """Fraction of the route's distance run on ground already covered.

        A little repetition is fine and sometimes unavoidable; a lot means the
        route is circling a few blocks rather than going somewhere, which is
        miserable to actually run.
        """
        seen: set[int] = set()
        repeated = 0.0
        total = 0.0
        for edge_id in self.edge_ids:
            length = self._edge_lengths.get(edge_id, 0.0)
            total += length
            if edge_id in seen:
                repeated += length
            seen.add(edge_id)
        return repeated / total if total > 0 else 0.0

    @property
    def loss_m(self) -> float:
        """Total descent in metres."""
        diffs = np.diff(self.elevations)
        return float(-diffs[diffs < 0].sum())


def _route_grades(
    route_dist: np.ndarray, route_elev: np.ndarray, window_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grade against fractional distance for a built route."""
    total = float(route_dist[-1])
    spacing = total / max(1, route_dist.size - 1)
    span = max(1, int(round(window_m / spacing)))
    if route_dist.size <= span:
        return np.empty(0), np.empty(0)
    rise = route_elev[span:] - route_elev[:-span]
    run = route_dist[span:] - route_dist[:-span]
    with np.errstate(invalid="ignore", divide="ignore"):
        grades = np.where(run > 0, rise / run * 100.0, 0.0)
    centres = (route_dist[span:] + route_dist[:-span]) / 2.0
    return centres / total, grades


def score_fit(route: Route) -> dict[str, float]:
    """Measure how closely a route reproduces the course profile.

    Compares grade at matching fractional positions, which is the quantity
    the route was built to match, and also reports the climbing totals, which
    it was not -- so the climbing figures are an independent check.

    Returns:
        A dictionary of fit statistics.
    """
    target = route.target
    # Compare at the course's own window, scaled to this route's length.
    window = target.window_m * (float(route.dist_m[-1]) / target.total_m)
    window = max(window, 100.0)

    fractions, grades = _route_grades(route.dist_m, route.elevations, window)
    if fractions.size == 0:
        return {}

    wanted = target.grade_at(fractions)
    differences = grades - wanted

    expected_gain = target.gain_for_length(float(route.dist_m[-1]))
    actual_gain = route.gain_m

    # Cumulative drift: how far the route sits from where the course would be
    # at the same fraction. Grade error is local and can look fine while this
    # grows without bound.
    total_m = float(route.dist_m[-1])
    length_scale = total_m / target.total_m
    route_fracs = route.dist_m / total_m
    route_delta = route.elevations - route.elevations[0]
    wanted_delta = target.elevation_delta_at(route_fracs, length_scale)
    drift = route_delta - wanted_delta

    return {
        "grade_mae_pct": float(np.mean(np.abs(differences))),
        "grade_rmse_pct": float(np.sqrt(np.mean(differences**2))),
        "grade_bias_pct": float(np.mean(differences)),
        "grade_correlation": (
            float(np.corrcoef(grades, wanted)[0, 1])
            if np.std(grades) > 1e-9 and np.std(wanted) > 1e-9
            else float("nan")
        ),
        "route_gain_ft": actual_gain * FEET_PER_METRE,
        "course_equivalent_gain_ft": expected_gain * FEET_PER_METRE,
        "gain_ratio": actual_gain / expected_gain if expected_gain > 0 else float("nan"),
        "steepest_climb_pct": float(np.max(grades)) if grades.size else 0.0,
        "steepest_descent_pct": float(np.min(grades)) if grades.size else 0.0,
        "course_steepest_climb_pct": float(np.max(target.grades_pct)),
        "course_steepest_descent_pct": float(np.min(target.grades_pct)),
        "mean_drift_ft": float(np.mean(np.abs(drift))) * FEET_PER_METRE,
        "max_drift_ft": float(np.max(np.abs(drift))) * FEET_PER_METRE,
    }


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@dataclass
class _Beam:
    """One partial route under consideration."""

    node: int
    distance_m: float
    cost: float
    edges: tuple[tuple[int, bool], ...]  # (edge_id, forward)
    recent: tuple[int, ...]
    elev_m: float = 0.0  # cumulative elevation change since the start


def _prune(
    beams: list[_Beam], width: int, per_node: int
) -> list[_Beam]:
    """Keep the cheapest beams, capping how many may sit on the same node.

    Without the per-node cap the whole beam collapses onto one street: every
    survivor is the same route with a different last turn, and the search
    stops exploring. Capping by node forces the beam to hold genuinely
    different routes.
    """
    beams.sort(key=lambda beam: beam.cost)
    kept: list[_Beam] = []
    seen: dict[int, int] = {}
    for beam in beams:
        count = seen.get(beam.node, 0)
        if count >= per_node:
            continue
        seen[beam.node] = count + 1
        kept.append(beam)
        if len(kept) >= width:
            break
    return kept


def build_route(
    graph: RoadGraph,
    target: CourseTarget,
    target_length_m: float,
    start_node: int,
    *,
    name: str = "route",
    beam_width: int = 220,
    recent_memory: int = 250,
    unpaved_penalty: float = 0.6,
    feature_emphasis: float = 0.6,
    drift_weight: float = 1.0,
    drift_scale_m: float = 10.0,
    close_loop: bool = True,
    milestone_m: float = 200.0,
    per_node: int = 2,
    length_tolerance: tuple[float, float] = (0.95, 1.08),
) -> Route | None:
    """Grow a route that tracks the course's grade profile.

    The search advances every candidate route to the same distance milestone
    before pruning. That synchronisation matters: cost accumulates with
    distance, so comparing a route that has run three miles against one that
    has run eight is meaningless, and ranking them together makes the search
    collapse onto whichever route is shortest rather than whichever best
    matches the course.

    At each milestone the cheapest ``beam_width`` routes survive, subject to a
    cap on how many may share an endpoint, which keeps the beam genuinely
    diverse rather than holding one route with many different last turns.

    Args:
        graph: The routable road graph.
        target: The course curve to track.
        target_length_m: How long the finished route should be.
        start_node: Node to start (and, for a loop, finish) at.
        name: Label for the finished route.
        beam_width: How many partial routes survive each milestone.
        recent_memory: How many recently used edges a beam refuses to reuse.
            This stops a route ping-ponging on one good hill while still
            permitting genuine laps of a long flat corridor.
        unpaved_penalty: Extra cost per metre of known-unpaved surface.
        feature_emphasis: How much harder to try where the course is steep.
            Error is multiplied by ``1 + feature_emphasis * |wanted grade|``,
            so failing to find a 5% pitch costs several times more than
            missing a half-percent roll. Without this the search sits on flat
            downtown streets: they match the course's many gentle miles
            cheaply, and the detour to a real hill never pays for itself.
        close_loop: Finish by routing back to ``start_node``.
        milestone_m: Distance between synchronisation points.
        per_node: Cap on surviving beams sharing an endpoint.
        length_tolerance: Acceptable finished length, as a fraction of the
            target, once the closing leg is included.

    Returns:
        The finished route, or ``None`` if the search died out.
    """
    home = graph.distances_from(start_node) if close_loop else {}
    if close_loop and len(home) < 2:
        LOG.warning("start node is isolated; cannot build a loop")
        return None

    low = target_length_m * length_tolerance[0]
    high = target_length_m * length_tolerance[1]
    # Elevation scales with length, which is what keeps grades unchanged when
    # a long course is reproduced over a shorter route.
    length_scale = target_length_m / target.total_m
    drift_divisor = max(1e-6, drift_scale_m)

    beams = [_Beam(start_node, 0.0, 0.0, (), ())]
    best: tuple[float, _Beam] | None = None
    milestones = int(math.ceil(target_length_m / milestone_m))

    for index in range(1, milestones + 2):
        milestone = min(index * milestone_m, target_length_m)
        working = beams

        # Advance every beam to the milestone, one edge at a time.
        for _ in range(64):
            if all(beam.distance_m >= milestone for beam in working):
                break
            grown: list[_Beam] = []
            for beam in working:
                if beam.distance_m >= milestone:
                    grown.append(beam)
                    continue
                for edge_id, neighbour in graph.adjacency.get(beam.node, ()):
                    if edge_id in beam.recent:
                        continue
                    routable = graph.edges[edge_id]
                    length = routable.length_m
                    if length <= 0.0:
                        continue
                    # Never wander so far that getting home blows the budget.
                    if close_loop:
                        budget = high - (beam.distance_m + length)
                        if home.get(neighbour, math.inf) > budget:
                            continue

                    forward = routable.edge.start_node == beam.node
                    grade = routable.grade_pct(forward)
                    midpoint = (
                        beam.distance_m + length / 2.0
                    ) / target_length_m
                    wanted = float(target.grade_at(midpoint))

                    emphasis = 1.0 + feature_emphasis * abs(wanted)
                    step_cost = length * abs(grade - wanted) * emphasis
                    if routable.edge.surface_class == "unpaved":
                        step_cost += length * unpaved_penalty

                    rise = routable.rise_m if forward else -routable.rise_m
                    new_elev = beam.elev_m + rise
                    if drift_weight > 0.0:
                        fraction_end = (
                            beam.distance_m + length
                        ) / target_length_m
                        wanted_elev = float(
                            target.elevation_delta_at(fraction_end, length_scale)
                        )
                        drift = abs(new_elev - wanted_elev)
                        step_cost += (
                            length * drift_weight * (drift / drift_divisor)
                        )

                    grown.append(
                        _Beam(
                            node=neighbour,
                            distance_m=beam.distance_m + length,
                            cost=beam.cost + step_cost,
                            edges=beam.edges + ((edge_id, forward),),
                            recent=(beam.recent + (edge_id,))[-recent_memory:],
                            elev_m=new_elev,
                        )
                    )
            if not grown:
                break
            working = _prune(grown, beam_width * 3, per_node * 2)

        if not working:
            break
        beams = _prune(list(working), beam_width, per_node)

        # Record any beam that could close at an acceptable total length.
        for beam in beams:
            if close_loop:
                to_home = home.get(beam.node, math.inf)
                total = beam.distance_m + to_home
            else:
                to_home = 0.0
                total = beam.distance_m
            if not (low <= total <= high):
                continue
            # Charge the closing leg at the average rate so a long detour home
            # is not free.
            rate = beam.cost / max(beam.distance_m, 1.0)
            score = (beam.cost + rate * to_home) / max(total, 1.0)
            if best is None or score < best[0]:
                best = (score, beam)

        if index % 20 == 0 and beams:
            LOG.info(
                "  %.2f mi, cost/m %.3f, %d beams",
                geo.metres_to_miles(beams[0].distance_m),
                beams[0].cost / max(beams[0].distance_m, 1.0),
                len(beams),
            )

    if best is None:
        LOG.warning(
            "no route closed within %.0f%%-%.0f%% of the target length",
            length_tolerance[0] * 100,
            length_tolerance[1] * 100,
        )
        return None

    finished = best[1]
    sequence = list(finished.edges)
    if close_loop and finished.node != start_node:
        for edge_id in graph.shortest_path_edges(finished.node, start_node):
            # Orientation is resolved when the geometry is stitched.
            sequence.append((edge_id, True))

    return _assemble_route(graph, sequence, start_node, name, target, close_loop)


def _assemble_route(
    graph: RoadGraph,
    sequence: Sequence[tuple[int, bool]],
    start_node: int,
    name: str,
    target: CourseTarget,
    close_loop: bool,
) -> Route | None:
    """Stitch an edge sequence into continuous geometry with elevation."""
    if not sequence:
        return None

    coords: list[np.ndarray] = []
    elevations: list[np.ndarray] = []
    street_names: list[str] = []
    surface_lengths: dict[str, float] = {"paved": 0.0, "unpaved": 0.0, "unknown": 0.0}
    cursor = start_node
    used: list[int] = []

    for edge_id, _ in sequence:
        routable = graph.edges[edge_id]
        edge = routable.edge
        if edge.start_node == cursor:
            forward = True
            cursor = edge.end_node
        elif edge.end_node == cursor:
            forward = False
            cursor = edge.start_node
        else:
            # The chain broke; stop cleanly rather than emit a teleport.
            break

        points, elevs = routable.traverse(forward)
        if coords:
            points, elevs = points[1:], elevs[1:]
        coords.append(points)
        elevations.append(elevs)
        used.append(edge_id)
        surface_lengths[edge.surface_class] += edge.length_m
        if edge.name and edge.name not in street_names:
            street_names.append(edge.name)

    if not coords:
        return None

    every_coord = np.vstack(coords)
    every_elevation = np.concatenate(elevations)
    if every_coord.shape[0] < 2:
        return None

    dist = geo.cumulative_distance_m(every_coord)
    total = float(sum(surface_lengths.values())) or 1.0

    route = Route(
        name=name,
        edge_ids=used,
        coords=every_coord,
        elevations=every_elevation,
        dist_m=dist,
        names=street_names,
        surface_fractions={
            key: value / total for key, value in surface_lengths.items()
        },
        closed_loop=close_loop and cursor == start_node,
        target=target,
        _edge_lengths={
            edge_id: graph.edges[edge_id].length_m for edge_id in set(used)
        },
    )
    route.fit = score_fit(route)
    return route


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_routes(
    config: Config,
    edges: Sequence[Edge],
    sampler,
    course_profile,
) -> list[Route]:
    """Build every route configured under ``routes`` in the config.

    Args:
        config: Loaded configuration.
        edges: Edges from :func:`analyze.build_edges`.
        sampler: An open :class:`analyze.DemSampler`.
        course_profile: A :class:`course.CourseProfile`.

    Returns:
        The routes that were successfully built.
    """
    specs = config.routes
    if not specs:
        return []

    target = course_target(
        course_profile,
        window_m=float(
            (config.analysis or {}).get("course_grade_window_m", 400.0)
        ),
    )

    graph = RoadGraph(
        edges,
        sampler,
        spacing_m=float(config.analysis.get("sample_spacing_m", 20.0)),
        smoothing_samples=int(config.analysis.get("smoothing_samples", 5)),
    )

    ref_lat, ref_lon = config.reference_point
    start_node = graph.nearest_node(ref_lat, ref_lon)
    LOG.info("starting routes from node %d near the reference point", start_node)

    routes: list[Route] = []
    for spec in specs:
        length_mi = float(spec.get("length_mi") or 0.0)
        if length_mi <= 0:
            length_m = target.total_m
            length_mi = geo.metres_to_miles(length_m)
        else:
            length_m = geo.miles_to_metres(length_mi)

        LOG.info(
            "building %s: %.2f mi %s",
            spec.get("key", "route"),
            length_mi,
            "loop" if spec.get("loop", True) else "point to point",
        )
        route = build_route(
            graph,
            target,
            length_m,
            start_node,
            name=str(spec.get("label", spec.get("key", "route"))),
            beam_width=int(spec.get("beam_width", 220)),
            recent_memory=int(spec.get("recent_memory", 250)),
            unpaved_penalty=float(spec.get("unpaved_penalty", 0.6)),
            feature_emphasis=float(spec.get("feature_emphasis", 0.6)),
            drift_weight=float(spec.get("drift_weight", 1.0)),
            per_node=int(spec.get("per_node", 2)),
            close_loop=bool(spec.get("loop", True)),
        )
        if route is None:
            LOG.warning("  could not build %s", spec.get("key"))
            continue

        route.name = str(spec.get("label", spec.get("key", "route")))
        LOG.info(
            "  %s: %.2f mi, %.0f ft gain (course rate %.0f ft), "
            "grade MAE %.2f%%, %s",
            route.name,
            route.length_mi,
            route.fit.get("route_gain_ft", 0.0),
            route.fit.get("course_equivalent_gain_ft", 0.0),
            route.fit.get("grade_mae_pct", float("nan")),
            "closed loop" if route.closed_loop else "not closed",
        )
        routes.append(route)

    return routes


# ---------------------------------------------------------------------------
# GPX output and command-line entry point
# ---------------------------------------------------------------------------


def write_gpx(route: Route, path: Path) -> Path:
    """Write a route as a single GPX track with elevation.

    Args:
        route: The route to write.
        path: Destination ``.gpx`` file.

    Returns:
        The path written.
    """
    from datetime import datetime, timezone
    from xml.sax.saxutils import escape as xml_escape

    fit = route.fit
    description = (
        f"{route.length_mi:.2f} mi, "
        f"{route.gain_m * FEET_PER_METRE:.0f} ft gain. "
        f"Built to track the race course's grade profile: "
        f"grade MAE {fit.get('grade_mae_pct', float('nan')):.2f}%, "
        f"correlation {fit.get('grade_correlation', float('nan')):.2f}. "
        f"Steepest sustained climb {fit.get('steepest_climb_pct', 0):.1f}%, "
        f"descent {fit.get('steepest_descent_pct', 0):.1f}%. "
        f"{'Closed loop.' if route.closed_loop else 'Point to point.'}"
    )

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="slo-terrain-matcher"'
        ' xmlns="http://www.topografix.com/GPX/1/1"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:schemaLocation="http://www.topografix.com/GPX/1/1'
        ' http://www.topografix.com/GPX/11.xsd">',
        "  <metadata>",
        f"    <name>{xml_escape(route.name)}</name>",
        f"    <desc>{xml_escape(description)}</desc>",
        f"    <time>{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}</time>",
        "  </metadata>",
        "  <trk>",
        f"    <name>{xml_escape(route.name)}</name>",
        f"    <desc>{xml_escape(description)}</desc>",
        "    <type>running</type>",
        "    <trkseg>",
    ]
    for (lon, lat), elevation in zip(route.coords, route.elevations):
        parts.append(
            f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}">'
            f"<ele>{float(elevation):.1f}</ele></trkpt>"
        )
    parts += ["    </trkseg>", "  </trk>", "</gpx>"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    LOG.info("wrote %s (%d points)", path.name, route.coords.shape[0])
    return path


def main(argv: list[str] | None = None) -> int:
    """Build one route of a given length and write it as GPX."""
    import argparse

    import analyze as analyze_module
    import course as course_module
    import fetch_data as fetch_module
    from config import configure_logging, load_config

    parser = argparse.ArgumentParser(description="Build a course-matching route.")
    parser.add_argument("--miles", type=float, required=True, help="route length")
    parser.add_argument("--out", default=None, help="output .gpx path")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--name", default=None, help="track name")
    parser.add_argument("--beam", type=int, default=600, help="beam width")
    parser.add_argument(
        "--emphasis",
        type=float,
        default=0.6,
        help="how hard to chase the course's steep sections",
    )
    parser.add_argument("--per-node", type=int, default=3, help="beam diversity cap")
    parser.add_argument(
        "--start",
        default=None,
        help="start point as 'lat,lon'; defaults to the configured reference point",
    )
    parser.add_argument(
        "--point-to-point",
        action="store_true",
        help="do not close the route back to the start",
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = load_config(args.config)
    if not config.course_gpx:
        parser.error("course_gpx must be set in config.yaml")

    dem_path = fetch_module.fetch_dem(config)
    osm_payload = fetch_module.fetch_osm(config)
    edges, _ = analyze_module.build_edges(osm_payload)
    sampler = analyze_module.DemSampler(dem_path)

    spacing = float(config.analysis.get("sample_spacing_m", 20.0))
    smoothing = int(config.analysis.get("smoothing_samples", 5))
    profile = course_module.load_course(Path(config.course_gpx), spacing, smoothing)
    target = course_target(profile)
    graph = RoadGraph(edges, sampler, spacing, smoothing)

    if args.start:
        lat_text, lon_text = args.start.split(",")
        start_lat, start_lon = float(lat_text), float(lon_text)
    else:
        start_lat, start_lon = config.reference_point
    start_node = graph.nearest_node(start_lat, start_lon)

    name = args.name or f"SLO {args.miles:g} mi - Oakland course match"
    route = build_route(
        graph,
        target,
        geo.miles_to_metres(args.miles),
        start_node,
        name=name,
        beam_width=args.beam,
        recent_memory=150,
        unpaved_penalty=0.9,
        feature_emphasis=args.emphasis,
        per_node=args.per_node,
        close_loop=not args.point_to_point,
        length_tolerance=(0.97, 1.03),
    )
    if route is None:
        LOG.error("could not build a route of that length from this start point")
        return 1

    out = Path(args.out) if args.out else (
        config.output_dir / f"route_{args.miles:g}mi.gpx"
    )
    write_gpx(route, out)

    fit = route.fit
    print()
    print(f"{route.name}")
    print(f"  length           {route.length_mi:.2f} mi")
    print(
        f"  climbing         {route.gain_m * FEET_PER_METRE:.0f} ft "
        f"(same distance of the race course: "
        f"{fit.get('course_equivalent_gain_ft', 0):.0f} ft)"
    )
    print(
        f"  steepest climb   {fit.get('steepest_climb_pct', 0):+.1f}%  "
        f"(course {fit.get('course_steepest_climb_pct', 0):+.1f}%)"
    )
    print(
        f"  steepest descent {fit.get('steepest_descent_pct', 0):+.1f}%  "
        f"(course {fit.get('course_steepest_descent_pct', 0):+.1f}%)"
    )
    print(f"  grade error      {fit.get('grade_mae_pct', 0):.2f} percentage points")
    print(f"  profile match    {fit.get('grade_correlation', 0):.2f} correlation")
    paved = route.surface_fractions.get("paved", 0.0) * 100
    unpaved = route.surface_fractions.get("unpaved", 0.0) * 100
    print(f"  surface          {paved:.0f}% known paved, {unpaved:.0f}% known unpaved")
    print(f"  closed loop      {'yes' if route.closed_loop else 'no'}")
    print(f"  repeated ground  {route.repeat_fraction * 100:.0f}% of the distance")
    print(f"  streets          {', '.join(route.names[:12])}")
    print()
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
