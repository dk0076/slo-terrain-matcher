"""Score how pleasant each stretch of road or path is to run on.

This exists because a route that reproduces the race profile perfectly can
still be a miserable run: half a mile of correct gradient along a four-lane
arterial is not training, it is an errand.

**It is not a popularity measure, and should not be read as one.** Strava's
heatmap would answer "do people actually run here", but their terms forbid
both scraping the tiles and storing anything derived from their API, so it is
not available to a tool that caches results. OpenStreetMap's public GPS trace
layer looks like a substitute and is not: it renders traces as opaque lines,
so it saturates on any road a car has driven. Measured against real streets it
rated a four-lane arterial above a well-used creek path.

So this scores *what a way is* rather than *who uses it*. A beautiful empty
path and a beloved one score alike. That is the honest limit of what open data
supports, and it is still enough to steer a route off the arterials and onto
the park paths.

Every signal comes from tags already fetched with the road network, except
greenspace, which is one extra Overpass query and is optional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from analyze import Edge

LOG = logging.getLogger("pleasant")

#: Neutral score. Everything is nudged up or down from here.
BASE = 0.5

#: Ways built for people on foot.
FOOT_FIRST = frozenset({"footway", "path", "pedestrian", "steps"})

#: Roads that carry enough traffic to spoil a run.
BUSY_ROADS = {
    "trunk": -0.34,
    "primary": -0.30,
    "trunk_link": -0.30,
    "primary_link": -0.26,
    "secondary": -0.18,
    "secondary_link": -0.16,
    "tertiary": -0.07,
    "tertiary_link": -0.07,
}

#: Quiet ways, and how much they help.
QUIET_WAYS = {
    "footway": 0.26,
    "path": 0.26,
    "pedestrian": 0.24,
    "cycleway": 0.22,
    "living_street": 0.14,
    "track": 0.10,
    "residential": 0.04,
    "unclassified": 0.02,
}

#: ``service`` covers parking aisles and alleys, which are dreary and often
#: dead ends. Driveways are worse.
SERVICE_PENALTY = -0.14
DRIVEWAY_PENALTY = -0.22

#: Speed above which a road stops being pleasant, in mph.
FAST_MPH = 45.0

#: Score given to a way that may not legally be run on. Not zero, because the
#: tags are sometimes wrong and a route may genuinely need one short link, but
#: low enough that the search will take almost any detour to avoid it.
BARRED = 0.02


@dataclass(frozen=True)
class Reason:
    """One contribution to a way's score, kept so the result can be explained."""

    label: str
    delta: float


def _parse_maxspeed_mph(value: str) -> float | None:
    """Read an OSM ``maxspeed`` tag as mph, or ``None`` if unparseable."""
    text = value.strip().lower()
    try:
        if text.endswith("mph"):
            return float(text[:-3].strip())
        return float(text) * 0.621371  # bare values are km/h
    except ValueError:
        return None


def score_edge(
    edge: Edge, in_greenspace: bool = False
) -> tuple[float, list[Reason]]:
    """Rate one edge from 0 (avoid) to 1 (lovely), with the reasons why.

    Args:
        edge: The edge to rate.
        in_greenspace: Whether it runs through a park or similar.

    Returns:
        ``(score, reasons)``. The score is clamped to ``[0, 1]``; the reasons
        explain how it got there, which matters because this is a judgement
        call rather than a measurement.
    """
    reasons: list[Reason] = []
    score = BASE
    highway = (edge.highway or "").lower()
    name = (edge.name or "").strip()

    def add(label: str, delta: float) -> None:
        nonlocal score
        if delta:
            score += delta
            reasons.append(Reason(label, delta))

    if highway in QUIET_WAYS:
        add(f"{highway}", QUIET_WAYS[highway])
    elif highway in BUSY_ROADS:
        add(f"{highway} road", BUSY_ROADS[highway])
    elif highway == "service":
        add("service road", SERVICE_PENALTY)

    if edge.service_tag == "driveway":
        add("driveway", DRIVEWAY_PENALTY)

    # A named path is one somebody thought worth naming, which is the closest
    # open data comes to "people go here".
    if name and highway in FOOT_FIRST | {"cycleway", "track"}:
        add("named trail", 0.13)

    if edge.foot_tag in {"designated", "yes"}:
        add("foot access", 0.09)

    if in_greenspace:
        add("through green space", 0.16)

    if edge.sidewalk_tag in {"both", "left", "right", "yes", "separate"}:
        add("has a sidewalk", 0.07)
    elif edge.sidewalk_tag == "no" and highway in BUSY_ROADS:
        add("no sidewalk on a busy road", -0.12)

    if edge.lit_tag == "yes":
        add("lit", 0.04)

    if edge.maxspeed_tag:
        mph = _parse_maxspeed_mph(edge.maxspeed_tag)
        if mph is not None and mph >= FAST_MPH:
            add(f"{mph:.0f} mph traffic", -0.12)

    if edge.lanes_tag:
        try:
            if int(float(edge.lanes_tag)) >= 4:
                add("four or more lanes", -0.10)
        except ValueError:
            pass

    if highway == "steps":
        add("steps", -0.25)

    score = max(0.0, min(1.0, score))

    # Access restrictions are not a penalty to be outweighed. A named park
    # path signposted "no pedestrians" is not a nice run, it is a trespass,
    # so being barred overrides every bonus rather than discounting them.
    if edge.foot_tag in {"no", "private"}:
        reasons.append(Reason("foot access forbidden", BARRED - score))
        score = BARRED
    elif edge.access_tag in {"private", "no"}:
        reasons.append(Reason("access restricted", BARRED - score))
        score = BARRED
    elif edge.access_tag == "customers":
        score = min(score, 0.25)

    return score, reasons


def greenspace_index(payload: dict[str, Any] | None):
    """Build a spatial index of green areas, or ``None`` if there are none.

    Args:
        payload: Overpass JSON from :func:`fetch_data.fetch_greenspace`.

    Returns:
        ``(tree, polygons)`` for :func:`edges_in_greenspace`, or ``None``.
    """
    if not payload:
        return None

    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    polygons = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry") or []
        coords = [
            (float(point["lon"]), float(point["lat"]))
            for point in geometry
            if point.get("lon") is not None
        ]
        if len(coords) >= 4:
            try:
                polygon = Polygon(coords)
                if polygon.is_valid and polygon.area > 0:
                    polygons.append(polygon)
            except Exception:  # noqa: BLE001 - skip malformed rings
                continue

    if not polygons:
        return None
    LOG.info("  %d green areas indexed", len(polygons))
    return STRtree(polygons), polygons


def edges_in_greenspace(
    edges: Sequence[Edge], index
) -> frozenset[int]:
    """Which edges have their midpoint inside a green area.

    Midpoint rather than full containment: a path along a park's edge is still
    a park path, and testing the whole geometry would reject it.
    """
    if index is None:
        return frozenset()

    from shapely.geometry import Point

    tree, polygons = index
    inside: set[int] = set()
    for edge in edges:
        if edge.coords.shape[0] < 2:
            continue
        middle = edge.coords[edge.coords.shape[0] // 2]
        point = Point(float(middle[0]), float(middle[1]))
        for position in tree.query(point):
            if polygons[int(position)].contains(point):
                inside.add(edge.edge_id)
                break

    LOG.info("  %d of %d edges run through green space", len(inside), len(edges))
    return frozenset(inside)


def score_network(
    edges: Sequence[Edge], greenspace: dict[str, Any] | None = None
) -> dict[int, float]:
    """Score every edge in the network.

    Args:
        edges: Edges from :func:`analyze.build_edges`.
        greenspace: Optional Overpass payload of parks and open space.

    Returns:
        A mapping of edge id to score in ``[0, 1]``.
    """
    index = greenspace_index(greenspace)
    green = edges_in_greenspace(edges, index)
    scores = {edge.edge_id: score_edge(edge, edge.edge_id in green)[0] for edge in edges}

    if scores:
        values = sorted(scores.values())
        LOG.info(
            "  pleasantness: median %.2f, worst %.2f, best %.2f",
            values[len(values) // 2],
            values[0],
            values[-1],
        )
    return scores
