"""Tests for the replica-versus-course comparison chart."""

from __future__ import annotations

import numpy as np
import pytest

import geo
from chart import DIVERGENCE_PCT, write_profile_comparison
from course import CourseProfile
from route import CourseTarget, Route


def _course(length_m: float = 42195.0, spacing_m: float = 20.0) -> CourseProfile:
    """A synthetic course that climbs gently then descends back."""
    count = int(length_m / spacing_m) + 1
    dist = np.linspace(0.0, length_m, count)
    elev = 50.0 + 30.0 * np.sin(np.pi * dist / length_m)
    lon = dist / 111_320.0
    points = np.column_stack([lon, np.full_like(lon, 35.0)])
    return CourseProfile(
        points=points, dist_m=dist, elev_m=elev, spacing_m=spacing_m
    )


def _target_from(profile: CourseProfile, window_m: float = 400.0) -> CourseTarget:
    span = max(1, int(round(window_m / profile.spacing_m)))
    rise = profile.elev_m[span:] - profile.elev_m[:-span]
    run = profile.dist_m[span:] - profile.dist_m[:-span]
    grades = np.where(run > 0, rise / run * 100.0, 0.0)
    centres = (profile.dist_m[span:] + profile.dist_m[:-span]) / 2.0
    diffs = np.diff(profile.elev_m)
    return CourseTarget(
        fractions=centres / float(profile.dist_m[-1]),
        grades_pct=grades,
        total_m=float(profile.dist_m[-1]),
        total_gain_m=float(diffs[diffs > 0].sum()),
        window_m=window_m,
    )


def _route(target: CourseTarget, length_m: float, elev_fn) -> Route:
    """A synthetic route of a given length with a supplied elevation shape."""
    spacing = 20.0
    count = int(length_m / spacing) + 1
    dist = np.linspace(0.0, length_m, count)
    elev = elev_fn(dist / length_m)
    lon = dist / 111_320.0
    coords = np.column_stack([lon, np.full_like(lon, 35.0)])
    return Route(
        name="test replica",
        edge_ids=[0],
        coords=coords,
        elevations=elev,
        dist_m=dist,
        names=["Test Street"],
        surface_fractions={"paved": 1.0, "unpaved": 0.0, "unknown": 0.0},
        closed_loop=True,
        target=target,
        fit={
            "grade_mae_pct": 0.5,
            "grade_correlation": 0.8,
            "course_equivalent_gain_ft": 100.0,
        },
        _edge_lengths={0: length_m},
    )


def test_chart_is_written_and_self_contained(tmp_path):
    """The chart must open with no network access and no plotting library."""
    profile = _course()
    target = _target_from(profile)
    route = _route(target, 14484.0, lambda f: 50.0 + 10.0 * np.sin(np.pi * f))

    path = write_profile_comparison(
        route, profile, tmp_path / "chart.html", course_name="test.gpx"
    )

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "<svg" in text and "polyline" in text
    assert "http://" not in text.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "<script" not in text


def test_chart_reports_both_lengths_and_the_fit(tmp_path):
    profile = _course()
    target = _target_from(profile)
    route = _route(target, 14484.0, lambda f: 50.0 + 10.0 * np.sin(np.pi * f))

    text = write_profile_comparison(
        route, profile, tmp_path / "c.html"
    ).read_text(encoding="utf-8")

    assert f"{route.length_mi:.2f} mi" in text
    assert f"{geo.metres_to_miles(target.total_m):.2f} mi" in text
    assert "0.80" in text or "0.8" in text


def test_a_matching_replica_shows_little_divergence(tmp_path):
    """A replica whose grades mirror the course flags almost nothing."""
    profile = _course()
    target = _target_from(profile)
    # Same shape, scaled by the length ratio: identical grades.
    length = 14484.0
    scale = length / target.total_m
    route = _route(
        target, length, lambda f: 50.0 + 30.0 * scale * np.sin(np.pi * f)
    )

    text = write_profile_comparison(
        route, profile, tmp_path / "c.html"
    ).read_text(encoding="utf-8")

    percent = int(text.split("Diverging stretches")[1].split("%")[0].split(">")[-1])
    assert percent <= 5, "a faithful replica should barely diverge"


def test_a_mismatched_replica_is_flagged(tmp_path):
    """A replica with the opposite profile is called out as diverging."""
    profile = _course()
    target = _target_from(profile)
    length = 14484.0
    # Inverted and exaggerated: descends where the course climbs.
    route = _route(
        target, length, lambda f: 50.0 - 120.0 * np.sin(np.pi * f)
    )

    text = write_profile_comparison(
        route, profile, tmp_path / "c.html"
    ).read_text(encoding="utf-8")

    percent = int(text.split("Diverging stretches")[1].split("%")[0].split(">")[-1])
    assert percent > 20, "an inverted replica should be flagged clearly"


def test_chart_refuses_a_route_too_short_to_measure(tmp_path):
    profile = _course()
    target = _target_from(profile)
    route = _route(target, 40.0, lambda f: np.full_like(f, 50.0))

    with pytest.raises(ValueError, match="too short"):
        write_profile_comparison(route, profile, tmp_path / "c.html")


def test_divergence_threshold_is_in_percentage_points():
    """The threshold is a grade difference, not a ratio."""
    assert 0.5 <= DIVERGENCE_PCT <= 5.0


def test_chart_escapes_names(tmp_path):
    """A course filename with markup must not break out into the page."""
    profile = _course()
    target = _target_from(profile)
    route = _route(target, 14484.0, lambda f: 50.0 + 10.0 * np.sin(np.pi * f))
    route.name = "<script>alert(1)</script>"

    text = write_profile_comparison(
        route, profile, tmp_path / "c.html", course_name="<b>x</b>.gpx"
    ).read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text
    assert "<b>x</b>.gpx" not in text
