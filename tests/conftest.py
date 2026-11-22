"""Shared fixtures and synthetic-profile builders for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The modules under test live at the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geo  # noqa: E402
from analyze import Corridor, Edge, Profile  # noqa: E402
from config import Target  # noqa: E402


def make_edge(
    edge_id: int = 0,
    start_node: int = 1,
    end_node: int = 2,
    length_m: float = 200.0,
    surface: str = "asphalt",
    name: str = "Test Street",
    bridge: bool = False,
    tunnel: bool = False,
    highway: str = "residential",
) -> Edge:
    """Build a straight east-west edge of a given length.

    The geometry is only used by surface and reliability logic in these tests,
    so a two-vertex line is enough.
    """
    # At the equator one degree of longitude is about 111.32 km.
    delta = length_m / 111_320.0
    coords = np.array([[0.0, 35.0], [delta, 35.0]], dtype=float)
    return Edge(
        edge_id=edge_id,
        way_id=edge_id,
        name=name,
        highway=highway,
        surface=surface,
        bridge=bridge,
        tunnel=tunnel,
        start_node=start_node,
        end_node=end_node,
        coords=coords,
        length_m=length_m,
    )


def make_corridor(edges: list[Edge] | None = None) -> Corridor:
    """Build a corridor from a list of edges laid end to end."""
    if edges is None:
        edges = [make_edge()]
    offsets = [0.0]
    for edge in edges:
        offsets.append(offsets[-1] + edge.length_m)
    total = offsets[-1]
    coords = np.array(
        [[0.0, 35.0], [total / 111_320.0, 35.0]], dtype=float
    )
    nodes = [edges[0].start_node] + [edge.end_node for edge in edges]
    return Corridor(
        corridor_id=0,
        edges=list(edges),
        coords=coords,
        node_sequence=nodes,
        node_offsets_m=np.array(offsets, dtype=float),
    )


def make_profile(
    grades_pct: list[tuple[float, float]],
    spacing_m: float = 20.0,
    start_elevation_m: float = 100.0,
    corridor: Corridor | None = None,
) -> Profile:
    """Build a profile with exactly known grades.

    Args:
        grades_pct: A list of ``(length_m, grade_pct)`` runs, applied in order.
        spacing_m: Sample spacing.
        start_elevation_m: Elevation of the first sample.
        corridor: Corridor to attach; a default one is built if omitted.

    Returns:
        A profile whose smoothed and raw elevations are identical, so grade
        assertions are exact.
    """
    distances = [0.0]
    elevations = [start_elevation_m]
    for length_m, grade in grades_pct:
        steps = int(round(length_m / spacing_m))
        for _ in range(steps):
            distances.append(distances[-1] + spacing_m)
            elevations.append(elevations[-1] + spacing_m * grade / 100.0)

    dist = np.array(distances, dtype=float)
    elev = np.array(elevations, dtype=float)
    # Lay the samples out along a line of latitude so geometry stays simple.
    lon = dist / 111_320.0
    points = np.column_stack([lon, np.full_like(lon, 35.0)])

    total = float(dist[-1])
    if corridor is None:
        corridor = make_corridor([make_edge(length_m=total)])

    return Profile(
        corridor=corridor,
        points=points,
        dist_m=dist,
        elev_raw_m=elev.copy(),
        elev_m=elev,
        spacing_m=spacing_m,
    )


def make_target(
    key: str = "test",
    length_mi: tuple[float, float] = (0.2, 0.5),
    grade_pct: tuple[float, float] = (4.0, 7.0),
    ideal_grade_pct: float | None = None,
    **kwargs,
) -> Target:
    """Build a target with sensible defaults for tests."""
    if ideal_grade_pct is None:
        ideal_grade_pct = sum(grade_pct) / 2.0
    return Target(
        key=key,
        label=kwargs.pop("label", key),
        race_location=kwargs.pop("race_location", "mile 1"),
        spec=kwargs.pop("spec", "test spec"),
        length_mi=length_mi,
        grade_pct=grade_pct,
        ideal_grade_pct=ideal_grade_pct,
        **kwargs,
    )


@pytest.fixture
def steady_climb() -> Profile:
    """A 1 km profile at a dead-steady 5% grade."""
    return make_profile([(1000.0, 5.0)])


@pytest.fixture
def mile() -> float:
    """Metres in a statute mile."""
    return geo.METRES_PER_MILE
