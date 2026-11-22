"""Tests for geodesy, smoothing and grade computation."""

from __future__ import annotations

import numpy as np
import pytest

import geo
from analyze import (
    PAVED_SURFACES,
    UNPAVED_SURFACES,
    rolling_extreme_grades,
    smooth_series,
)
from conftest import make_corridor, make_edge, make_profile


# ---------------------------------------------------------------------------
# Geodesy
# ---------------------------------------------------------------------------


def test_haversine_known_distance():
    """One degree of latitude is about 111.2 km anywhere on the sphere."""
    distance = geo.haversine_m(35.0, -120.0, 36.0, -120.0)
    assert 111_000 < distance < 111_400


def test_haversine_is_symmetric_and_zero_at_a_point():
    assert geo.haversine_m(35.0, -120.0, 35.0, -120.0) == pytest.approx(0.0)
    there = geo.haversine_m(35.0, -120.0, 35.1, -120.1)
    back = geo.haversine_m(35.1, -120.1, 35.0, -120.0)
    assert there == pytest.approx(back)


def test_cumulative_distance_is_monotonic_and_matches_haversine():
    coords = np.array(
        [[-120.60, 35.28], [-120.59, 35.29], [-120.58, 35.30]], dtype=float
    )
    along = geo.cumulative_distance_m(coords)
    assert along[0] == 0.0
    assert np.all(np.diff(along) > 0)
    direct = geo.haversine_m(35.28, -120.60, 35.29, -120.59)
    assert along[1] == pytest.approx(direct, rel=1e-9)


def test_cumulative_distance_handles_degenerate_input():
    assert geo.cumulative_distance_m(np.empty((0, 2))).size == 0
    assert geo.cumulative_distance_m(np.array([[0.0, 0.0]])).tolist() == [0.0]


def test_resample_polyline_spacing_and_endpoints():
    """Resampling keeps both ends and produces near-uniform spacing."""
    coords = np.array([[0.0, 35.0], [0.02, 35.0]], dtype=float)
    total = geo.cumulative_distance_m(coords)[-1]
    points, dist = geo.resample_polyline(coords, 20.0)

    assert dist[0] == 0.0
    assert dist[-1] == pytest.approx(total)
    assert points[0] == pytest.approx(coords[0])
    assert points[-1] == pytest.approx(coords[-1])
    steps = np.diff(dist)
    assert np.allclose(steps, steps[0])
    assert steps[0] <= 20.0 + 1e-9


def test_resample_polyline_rejects_bad_spacing():
    coords = np.array([[0.0, 35.0], [0.01, 35.0]], dtype=float)
    with pytest.raises(ValueError):
        geo.resample_polyline(coords, 0.0)


def test_bearing_cardinal_directions():
    assert geo.bearing_deg(35.0, -120.0, 36.0, -120.0) == pytest.approx(0.0, abs=1e-6)
    assert geo.bearing_deg(35.0, -120.0, 35.0, -119.0) == pytest.approx(
        90.0, abs=0.5
    )
    assert geo.bearing_deg(35.0, -120.0, 34.0, -120.0) == pytest.approx(
        180.0, abs=1e-6
    )


def test_turn_angle_wraps_correctly():
    assert geo.turn_angle_deg(10.0, 20.0) == pytest.approx(10.0)
    assert geo.turn_angle_deg(350.0, 10.0) == pytest.approx(20.0)
    assert geo.turn_angle_deg(10.0, 350.0) == pytest.approx(20.0)
    assert geo.turn_angle_deg(0.0, 180.0) == pytest.approx(180.0)
    assert 0.0 <= geo.turn_angle_deg(123.0, 312.0) <= 180.0


def test_mile_conversions_round_trip():
    assert geo.metres_to_miles(geo.miles_to_metres(3.5)) == pytest.approx(3.5)
    assert geo.miles_to_metres(1.0) == pytest.approx(1609.344)


def test_google_maps_link_contains_coordinates():
    link = geo.google_maps_link(35.2828, -120.6596)
    assert "35.282800" in link and "-120.659600" in link
    assert link.startswith("https://")


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_smoothing_preserves_a_straight_line():
    """A constant-grade ramp must survive smoothing unchanged.

    This is the property that matters: smoothing is there to kill noise, and
    it must not bend a genuine steady climb.
    """
    values = np.arange(0.0, 50.0, 1.0)
    smoothed = smooth_series(values, 5)
    assert smoothed.size == values.size
    # Interior points are exactly preserved; the edge-padded ends are not.
    assert np.allclose(smoothed[2:-2], values[2:-2])


def test_smoothing_suppresses_a_single_spike():
    values = np.zeros(21)
    values[10] = 10.0
    smoothed = smooth_series(values, 5)
    assert smoothed[10] == pytest.approx(2.0)
    assert smoothed.max() == pytest.approx(2.0)


def test_smoothing_is_a_no_op_for_small_windows():
    values = np.array([1.0, 5.0, 2.0])
    assert np.array_equal(smooth_series(values, 1), values)
    assert np.array_equal(smooth_series(values, 0), values)


def test_smoothing_rounds_even_windows_up_to_stay_centred():
    values = np.zeros(21)
    values[10] = 10.0
    assert np.allclose(smooth_series(values, 4), smooth_series(values, 5))


def test_smoothing_ignores_nan():
    values = np.array([0.0, np.nan, 0.0, 0.0, 0.0])
    smoothed = smooth_series(values, 3)
    assert np.isfinite(smoothed).all()
    assert smoothed[2] == pytest.approx(0.0)


def test_smoothing_handles_empty_and_tiny_arrays():
    assert smooth_series(np.empty(0), 5).size == 0
    assert smooth_series(np.array([3.0]), 5).tolist() == [3.0]


# ---------------------------------------------------------------------------
# Grade computation
# ---------------------------------------------------------------------------


def test_synthetic_steady_climb_has_the_grade_it_was_built_with(steady_climb):
    """A profile built at 5% must measure 5% end to end."""
    rise = steady_climb.elev_m[-1] - steady_climb.elev_m[0]
    run = steady_climb.dist_m[-1] - steady_climb.dist_m[0]
    assert rise / run * 100.0 == pytest.approx(5.0)


@pytest.mark.parametrize("grade", [-8.0, -5.1, -1.25, 0.0, 1.5, 6.9, 12.0])
def test_profile_builder_reproduces_any_grade(grade):
    profile = make_profile([(400.0, grade)])
    rise = profile.elev_m[-1] - profile.elev_m[0]
    run = profile.dist_m[-1] - profile.dist_m[0]
    assert rise / run * 100.0 == pytest.approx(grade)


def test_rolling_extreme_grades_finds_the_steep_section():
    """A 10% ramp buried in flat ground is found at the 100 m scale."""
    profile = make_profile([(300.0, 0.0), (200.0, 10.0), (300.0, 0.0)])
    starts, grades = rolling_extreme_grades(profile, min_window_m=100.0)

    assert starts.size == grades.size > 0
    assert float(np.nanmax(grades)) == pytest.approx(10.0)
    assert float(np.nanmin(grades)) == pytest.approx(0.0)
    # The steepest 100 m window starts inside the ramp.
    steepest_at = float(starts[int(np.nanargmax(grades))])
    assert 300.0 <= steepest_at <= 400.0


def test_rolling_extreme_grades_averages_over_the_window():
    """A 20 m spike is diluted, not reported as a wall.

    This is the whole reason max grade is defined over a window: a single
    DEM cell pair must not be able to claim a 50% pitch.
    """
    profile = make_profile([(200.0, 0.0), (20.0, 50.0), (200.0, 0.0)])
    _, grades = rolling_extreme_grades(profile, min_window_m=100.0)
    assert float(np.nanmax(grades)) == pytest.approx(10.0)


def test_rolling_extreme_grades_on_descent_is_negative():
    profile = make_profile([(400.0, -5.1)])
    _, grades = rolling_extreme_grades(profile, min_window_m=100.0)
    assert float(np.nanmin(grades)) == pytest.approx(-5.1)
    assert float(np.nanmax(grades)) == pytest.approx(-5.1)


def test_rolling_extreme_grades_empty_when_profile_too_short():
    profile = make_profile([(40.0, 5.0)])
    starts, grades = rolling_extreme_grades(profile, min_window_m=100.0)
    assert starts.size == 0 and grades.size == 0


# ---------------------------------------------------------------------------
# Surface classification
# ---------------------------------------------------------------------------


def test_surface_classes_are_disjoint():
    assert not (PAVED_SURFACES & UNPAVED_SURFACES)


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("asphalt", "paved"),
        ("concrete", "paved"),
        ("paving_stones", "paved"),
        ("gravel", "unpaved"),
        ("dirt", "unpaved"),
        ("ground", "unpaved"),
        ("", "unknown"),
        ("something_new", "unknown"),
    ],
)
def test_edge_surface_classification(tag, expected):
    assert make_edge(surface=tag).surface_class == expected


def test_absent_surface_tag_is_unknown_not_paved():
    """OSM surface tags are incomplete; absent must never mean paved."""
    edge = make_edge(surface="", highway="residential")
    assert edge.surface_class == "unknown"


# ---------------------------------------------------------------------------
# Edge geometry
# ---------------------------------------------------------------------------


def test_edge_end_bearing_points_the_right_way():
    """An east-running edge arrives heading east, and west when reversed."""
    edge = make_edge(start_node=1, end_node=2, length_m=500.0)
    assert edge.end_bearing(2) == pytest.approx(90.0, abs=0.5)
    assert edge.end_bearing(1) == pytest.approx(270.0, abs=0.5)


def test_edge_end_bearing_rejects_a_foreign_node():
    with pytest.raises(ValueError):
        make_edge(start_node=1, end_node=2).end_bearing(99)


def test_corridor_length_and_offsets_agree():
    edges = [
        make_edge(edge_id=0, start_node=1, end_node=2, length_m=300.0),
        make_edge(edge_id=1, start_node=2, end_node=3, length_m=200.0),
    ]
    corridor = make_corridor(edges)
    assert corridor.length_m == pytest.approx(500.0)
    assert corridor.node_offsets_m.tolist() == [0.0, 300.0, 500.0]
    assert corridor.node_sequence == [1, 2, 3]


def test_corridor_edges_between_selects_overlapping_edges():
    edges = [
        make_edge(edge_id=0, start_node=1, end_node=2, length_m=300.0),
        make_edge(edge_id=1, start_node=2, end_node=3, length_m=300.0),
        make_edge(edge_id=2, start_node=3, end_node=4, length_m=300.0),
    ]
    corridor = make_corridor(edges)
    assert [e.edge_id for e in corridor.edges_between(0.0, 250.0)] == [0]
    assert [e.edge_id for e in corridor.edges_between(250.0, 350.0)] == [0, 1]
    assert [e.edge_id for e in corridor.edges_between(0.0, 900.0)] == [0, 1, 2]


def test_corridor_label_prefers_street_names():
    edges = [
        make_edge(edge_id=0, start_node=1, end_node=2, name="Peach Street"),
        make_edge(edge_id=1, start_node=2, end_node=3, name="Peach Street"),
    ]
    assert make_corridor(edges).label() == "Peach Street"

    unnamed = [make_edge(edge_id=0, name="", highway="track")]
    assert "track" in make_corridor(unnamed).label()
