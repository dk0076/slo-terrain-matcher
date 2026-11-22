"""Tests for the DEM validation step.

The FIT *parsing* needs a real file and is exercised by running
``validate.py`` against your own runs. Everything downstream of parsing is
covered here with a synthetic track, including a run against the real cached
DEM when one is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import geo
from config import load_config
from validate import FitTrack, ValidationReport, _grades, validate_track

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Windowed grade helper
# ---------------------------------------------------------------------------


def test_grades_over_windows_match_a_known_ramp():
    """A steady 5% ramp reads 5% in every window."""
    dist = np.arange(0.0, 1000.0, 10.0)
    elev = dist * 0.05

    midpoints, grades = _grades(dist, elev, window_m=100.0)

    assert grades.size > 0
    assert np.allclose(grades, 5.0)
    assert midpoints.size == grades.size


def test_grades_windows_are_non_overlapping_and_ordered():
    dist = np.arange(0.0, 1000.0, 10.0)
    elev = np.zeros_like(dist)

    midpoints, _ = _grades(dist, elev, window_m=100.0)

    assert np.all(np.diff(midpoints) > 0)
    assert np.all(np.diff(midpoints) >= 90.0)


def test_grades_sign_follows_the_terrain():
    dist = np.arange(0.0, 600.0, 10.0)
    _, down = _grades(dist, -dist * 0.05, 100.0)
    _, up = _grades(dist, dist * 0.05, 100.0)

    assert np.all(down < 0) and np.all(up > 0)


def test_grades_returns_nothing_for_a_track_shorter_than_one_window():
    dist = np.array([0.0, 10.0, 20.0])
    midpoints, grades = _grades(dist, dist * 0.05, window_m=100.0)
    assert midpoints.size == 0 and grades.size == 0


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _report(mae: float, correlation: float) -> ValidationReport:
    """Build a report carrying only the fields the verdict reads."""
    track = FitTrack(
        path=Path("synthetic.fit"),
        coords=np.zeros((2, 2)),
        elev_m=np.zeros(2),
        dist_m=np.array([0.0, 100.0]),
        altitude_source="test",
    )
    return ValidationReport(
        track=track,
        sample_count=2,
        coverage=1.0,
        mean_offset_m=0.0,
        median_offset_m=0.0,
        offset_stdev_m=0.0,
        elevation_rmse_m=0.0,
        detrended_rmse_m=0.0,
        grade_window_m=100.0,
        grade_bias_pct=0.0,
        grade_mae_pct=mae,
        grade_rmse_pct=mae,
        grade_correlation=correlation,
        fit_gain_m=0.0,
        dem_gain_m=0.0,
    )


def test_verdict_is_positive_for_tight_agreement():
    assert "trustworthy" in _report(0.3, 0.95).verdict()


def test_verdict_is_cautious_in_the_middle():
    assert "usable but loose" in _report(0.8, 0.80).verdict()


def test_verdict_is_negative_for_poor_agreement():
    assert "disagree" in _report(3.0, 0.2).verdict()


def test_summary_mentions_the_file_and_the_altitude_source():
    summary = _report(0.3, 0.95).summary()
    assert "synthetic.fit" in summary
    assert "Grade over 100 m windows" in summary


# ---------------------------------------------------------------------------
# End to end against the real DEM
# ---------------------------------------------------------------------------


def _cached_dem() -> Path | None:
    """The cached DEM clip, if ``fetch_data.py`` has been run."""
    candidates = sorted((ROOT / "data" / "dem").glob("dem_*.tif"))
    return candidates[0] if candidates else None


@pytest.mark.skipif(_cached_dem() is None, reason="no cached DEM; run fetch_data.py")
def test_validate_track_against_the_real_dem():
    """A track whose elevation *is* the DEM must validate near-perfectly.

    This exercises everything downstream of FIT parsing: DEM sampling,
    smoothing, offset statistics and the grade comparison.
    """
    from analyze import DemSampler

    dem_path = _cached_dem()
    config = load_config(ROOT / "config.yaml")

    # A straight line across the hills north-east of downtown SLO.
    lons = np.linspace(-120.660, -120.630, 400)
    lats = np.linspace(35.283, 35.298, 400)
    coords = np.column_stack([lons, lats])

    sampler = DemSampler(dem_path)
    elevations = sampler.sample(lons, lats)
    assert np.isfinite(elevations).all(), "test line must lie inside the DEM"

    track = FitTrack(
        path=Path("synthetic.fit"),
        coords=coords,
        elev_m=elevations,
        dist_m=geo.cumulative_distance_m(coords),
        altitude_source="synthetic (copied from the DEM)",
    )

    report = validate_track(track, dem_path, config)

    assert report.coverage == pytest.approx(1.0)
    assert report.mean_offset_m == pytest.approx(0.0, abs=1e-6)
    assert report.elevation_rmse_m == pytest.approx(0.0, abs=1e-6)
    assert report.grade_mae_pct == pytest.approx(0.0, abs=1e-6)
    assert "trustworthy" in report.verdict()


@pytest.mark.skipif(_cached_dem() is None, reason="no cached DEM; run fetch_data.py")
def test_validate_track_detects_a_constant_offset():
    """A barometer reading 12 m high shows up as offset, not as grade error.

    This is the distinction the whole step exists to make: a datum offset is
    harmless, while a grade error is not.
    """
    from analyze import DemSampler

    dem_path = _cached_dem()
    config = load_config(ROOT / "config.yaml")

    lons = np.linspace(-120.660, -120.630, 400)
    lats = np.linspace(35.283, 35.298, 400)
    coords = np.column_stack([lons, lats])

    sampler = DemSampler(dem_path)
    elevations = sampler.sample(lons, lats) + 12.0

    track = FitTrack(
        path=Path("offset.fit"),
        coords=coords,
        elev_m=elevations,
        dist_m=geo.cumulative_distance_m(coords),
        altitude_source="synthetic",
    )

    report = validate_track(track, dem_path, config)

    assert report.mean_offset_m == pytest.approx(12.0, abs=1e-6)
    assert report.detrended_rmse_m == pytest.approx(0.0, abs=1e-6)
    assert report.grade_mae_pct == pytest.approx(0.0, abs=1e-6)
    assert "trustworthy" in report.verdict()


@pytest.mark.skipif(_cached_dem() is None, reason="no cached DEM; run fetch_data.py")
def test_validate_track_rejects_a_track_outside_the_dem():
    dem_path = _cached_dem()
    config = load_config(ROOT / "config.yaml")

    # Somewhere in the Atlantic.
    coords = np.column_stack(
        [np.linspace(-30.0, -29.99, 50), np.linspace(40.0, 40.01, 50)]
    )
    track = FitTrack(
        path=Path("elsewhere.fit"),
        coords=coords,
        elev_m=np.zeros(50),
        dist_m=geo.cumulative_distance_m(coords),
        altitude_source="synthetic",
    )

    with pytest.raises(ValueError, match="falls inside the DEM"):
        validate_track(track, dem_path, config)
