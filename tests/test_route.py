"""Tests for the course target, the drift correction and route scoring."""

from __future__ import annotations

import numpy as np
import pytest

from course import CourseProfile
from route import CourseTarget, Route, course_target, score_fit


def _profile(length_m: float = 42195.0, spacing_m: float = 20.0) -> CourseProfile:
    """A course that climbs to a mid-point summit and comes back down."""
    count = int(length_m / spacing_m) + 1
    dist = np.linspace(0.0, length_m, count)
    elev = 100.0 + 40.0 * np.sin(np.pi * dist / length_m)
    lon = dist / 111_320.0
    points = np.column_stack([lon, np.full_like(lon, 35.0)])
    return CourseProfile(points=points, dist_m=dist, elev_m=elev, spacing_m=spacing_m)


def _route_from(target: CourseTarget, length_m: float, elev: np.ndarray) -> Route:
    dist = np.linspace(0.0, length_m, elev.size)
    lon = dist / 111_320.0
    coords = np.column_stack([lon, np.full_like(lon, 35.0)])
    return Route(
        name="r",
        edge_ids=[0],
        coords=coords,
        elevations=elev,
        dist_m=dist,
        names=[],
        surface_fractions={"paved": 1.0, "unpaved": 0.0, "unknown": 0.0},
        closed_loop=True,
        target=target,
        _edge_lengths={0: length_m},
    )


# ---------------------------------------------------------------------------
# The course's elevation curve
# ---------------------------------------------------------------------------


def test_course_target_carries_an_elevation_curve():
    target = course_target(_profile())
    assert target.elev_fractions.size > 2
    assert target.elev_delta_m.size == target.elev_fractions.size


def test_elevation_delta_starts_at_zero():
    """The curve is relative to the course's own start, not sea level."""
    target = course_target(_profile())
    assert float(target.elevation_delta_at(0.0)) == pytest.approx(0.0, abs=1e-6)


def test_elevation_delta_peaks_in_the_middle_for_a_summit_course():
    target = course_target(_profile())
    middle = float(target.elevation_delta_at(0.5))
    assert middle == pytest.approx(40.0, abs=1.0)
    assert middle > float(target.elevation_delta_at(0.1))
    assert middle > float(target.elevation_delta_at(0.9))


def test_elevation_delta_scales_with_length():
    """Compressing the course to a third keeps grades, so elevation thirds.

    This is what makes a short replica comparable to a long course.
    """
    target = course_target(_profile())
    full = float(target.elevation_delta_at(0.5, 1.0))
    third = float(target.elevation_delta_at(0.5, 1.0 / 3.0))
    assert third == pytest.approx(full / 3.0, rel=1e-6)


def test_elevation_delta_clamps_outside_the_range():
    target = course_target(_profile())
    assert float(target.elevation_delta_at(-0.5)) == pytest.approx(
        float(target.elevation_delta_at(0.0))
    )
    assert float(target.elevation_delta_at(1.5)) == pytest.approx(
        float(target.elevation_delta_at(1.0))
    )


def test_elevation_delta_accepts_arrays():
    target = course_target(_profile())
    values = target.elevation_delta_at(np.array([0.0, 0.5, 1.0]))
    assert values.shape == (3,)


# ---------------------------------------------------------------------------
# Drift reporting
# ---------------------------------------------------------------------------


def test_a_route_following_the_course_has_no_drift():
    """A replica tracing the course's own shape reports drift near zero."""
    profile = _profile()
    target = course_target(profile)
    length = target.total_m / 3.0
    scale = length / target.total_m

    fractions = np.linspace(0.0, 1.0, 500)
    elev = 10.0 + target.elevation_delta_at(fractions, scale)
    fit = score_fit(_route_from(target, length, elev))

    assert fit["mean_drift_ft"] == pytest.approx(0.0, abs=1.0)
    assert fit["max_drift_ft"] == pytest.approx(0.0, abs=2.0)


def test_a_route_that_sags_reports_drift():
    """A replica that drops below the course is caught even if grades look fine."""
    profile = _profile()
    target = course_target(profile)
    length = target.total_m / 3.0

    fractions = np.linspace(0.0, 1.0, 500)
    # Dips 20 m below its start in the middle, where the course climbs.
    elev = 10.0 - 20.0 * np.sin(np.pi * fractions)
    fit = score_fit(_route_from(target, length, elev))

    assert fit["max_drift_ft"] > 60.0
    assert fit["mean_drift_ft"] > 20.0


def test_a_constant_offset_is_not_drift():
    """Starting higher than the course is not drift; both are relative.

    The replica and the course are each measured from their own start, so a
    route that simply sits 50 m up a hill is a faithful replica, not a
    drifting one.
    """
    profile = _profile()
    target = course_target(profile)
    length = target.total_m / 3.0
    scale = length / target.total_m

    fractions = np.linspace(0.0, 1.0, 500)
    elev = 500.0 + target.elevation_delta_at(fractions, scale)
    fit = score_fit(_route_from(target, length, elev))

    assert fit["max_drift_ft"] == pytest.approx(0.0, abs=1.0)


def test_drift_is_reported_in_feet():
    """A replica ending 10 m above the course reports about 33 ft of drift."""
    profile = _profile()
    target = course_target(profile)
    length = target.total_m / 3.0
    scale = length / target.total_m

    fractions = np.linspace(0.0, 1.0, 500)
    # Pulls steadily away, reaching 10 m above the course by the finish.
    elev = 10.0 + target.elevation_delta_at(fractions, scale) + 10.0 * fractions
    fit = score_fit(_route_from(target, length, elev))

    assert fit["max_drift_ft"] == pytest.approx(32.8, abs=2.0)
    assert fit["mean_drift_ft"] == pytest.approx(16.4, abs=2.0)


def test_score_fit_reports_both_drift_measures():
    profile = _profile()
    target = course_target(profile)
    fractions = np.linspace(0.0, 1.0, 300)
    elev = 10.0 + 5.0 * np.sin(2 * np.pi * fractions)
    fit = score_fit(_route_from(target, target.total_m / 4.0, elev))
    assert "mean_drift_ft" in fit and "max_drift_ft" in fit
    assert fit["max_drift_ft"] >= fit["mean_drift_ft"]


# ---------------------------------------------------------------------------
# Gain scaling
# ---------------------------------------------------------------------------


def test_gain_for_length_scales_linearly():
    target = course_target(_profile())
    full = target.gain_for_length(target.total_m)
    half = target.gain_for_length(target.total_m / 2.0)
    assert half == pytest.approx(full / 2.0)
    assert full == pytest.approx(target.total_gain_m)
