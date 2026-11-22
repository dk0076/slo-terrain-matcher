"""Re-measure the configured targets against the race course GPX.

The target bands in ``config.yaml`` were written down from the Oakland
Marathon course. This module reads the course GPX back and reports what the
course actually does at each target's race location, so a band that has
drifted away from the course can be spotted and corrected.

Usage::

    python course.py path/to/course.gpx
    python course.py path/to/course.gpx --feature bridge_climb
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import geo
from config import Config, Target, configure_logging, load_config

LOG = logging.getLogger("course")

#: Mile ranges are written as free text in the config; pull numbers out of it.
_MILE_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)")
_SINGLE_MILE = re.compile(r"(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class CourseProfile:
    """A race course resampled to even spacing with smoothed elevation."""

    points: np.ndarray  # (n, 2) of (lon, lat)
    dist_m: np.ndarray
    elev_m: np.ndarray
    spacing_m: float

    @property
    def total_mi(self) -> float:
        """Course length in miles."""
        return geo.metres_to_miles(float(self.dist_m[-1]))

    def grade_between(self, start_mi: float, end_mi: float) -> float:
        """Mean grade over a mile range, in percent."""
        start_m = geo.miles_to_metres(start_mi)
        end_m = geo.miles_to_metres(end_mi)
        i = int(np.searchsorted(self.dist_m, start_m))
        j = int(np.searchsorted(self.dist_m, end_m))
        i = min(max(i, 0), self.dist_m.size - 1)
        j = min(max(j, i + 1), self.dist_m.size - 1)
        run = float(self.dist_m[j] - self.dist_m[i])
        if run <= 0:
            return 0.0
        return float(self.elev_m[j] - self.elev_m[i]) / run * 100.0

    def best_window(
        self,
        start_mi: float,
        end_mi: float,
        window_range_mi: tuple[float, float],
        steepest: bool,
    ) -> tuple[float, float, float, float]:
        """Find the steepest or flattest window inside a mile range.

        Every window length in ``window_range_mi`` is tried, not just one.
        This matters: grade falls off sharply with window length, so a pitch
        that reads +6.9% over a fifth of a mile reads under +5% if measured
        over half a mile. Sweeping the band finds the length at which the
        feature is actually most pronounced, which is the scale the written
        specification refers to.

        Args:
            start_mi: Start of the range to search, miles.
            end_mi: End of the range to search, miles.
            window_range_mi: ``(minimum, maximum)`` window length, miles.
            steepest: Look for the largest absolute grade when True, the
                smallest when False.

        Returns:
            ``(grade_pct, window_start_mi, window_end_mi, window_length_mi)``.
        """
        lo = int(np.searchsorted(self.dist_m, geo.miles_to_metres(start_mi)))
        hi = min(
            int(np.searchsorted(self.dist_m, geo.miles_to_metres(end_mi))),
            self.dist_m.size - 1,
        )

        span_lo = max(
            1, int(round(geo.miles_to_metres(window_range_mi[0]) / self.spacing_m))
        )
        span_hi = max(
            span_lo,
            int(round(geo.miles_to_metres(window_range_mi[1]) / self.spacing_m)),
        )
        span_hi = min(span_hi, max(span_lo, hi - lo))

        best: tuple[float, int, int] | None = None
        for span in range(span_lo, span_hi + 1):
            if hi - lo < span:
                break
            for start in range(lo, hi - span + 1):
                end = start + span
                run = float(self.dist_m[end] - self.dist_m[start])
                if run <= 0:
                    continue
                grade = float(self.elev_m[end] - self.elev_m[start]) / run * 100.0
                if best is None:
                    best = (grade, start, end)
                    continue
                better = (
                    abs(grade) > abs(best[0])
                    if steepest
                    else abs(grade) < abs(best[0])
                )
                if better:
                    best = (grade, start, end)

        if best is None:
            grade = self.grade_between(start_mi, end_mi)
            return grade, start_mi, end_mi, end_mi - start_mi

        grade, start, end = best
        return (
            grade,
            geo.metres_to_miles(float(self.dist_m[start])),
            geo.metres_to_miles(float(self.dist_m[end])),
            geo.metres_to_miles(float(self.dist_m[end] - self.dist_m[start])),
        )


def load_course(
    gpx_path: Path, spacing_m: float = 20.0, smoothing_samples: int = 5
) -> CourseProfile:
    """Read a course GPX and resample it the same way road corridors are.

    Using the same spacing and smoothing as the road analysis is what makes
    the two sets of numbers comparable; a raw GPX profile differenced at its
    native point spacing reads much steeper than a smoothed 20 m one.

    Args:
        gpx_path: Path to a GPX file with at least one track.
        spacing_m: Resample spacing, metres.
        smoothing_samples: Rolling-mean window applied to elevation.

    Returns:
        The resampled course profile.

    Raises:
        FileNotFoundError: If the GPX file does not exist.
        ValueError: If the file has no track points or no elevation data.
    """
    import gpxpy

    from analyze import smooth_series

    gpx_path = Path(gpx_path)
    if not gpx_path.exists():
        raise FileNotFoundError(f"GPX not found: {gpx_path}")

    with gpx_path.open("r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)

    coords: list[tuple[float, float]] = []
    elevations: list[float] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append((float(point.longitude), float(point.latitude)))
                elevations.append(
                    float(point.elevation) if point.elevation is not None else np.nan
                )

    if len(coords) < 2:
        raise ValueError(f"{gpx_path.name} has fewer than two track points")
    raw_elev = np.array(elevations, dtype=float)
    if not np.isfinite(raw_elev).any():
        raise ValueError(f"{gpx_path.name} has no elevation data")

    track_coords = np.array(coords, dtype=float)
    along = geo.cumulative_distance_m(track_coords)

    finite = np.isfinite(raw_elev)
    if not finite.all():
        raw_elev = np.interp(along, along[finite], raw_elev[finite])

    points, dist = geo.resample_polyline(track_coords, spacing_m)
    elev = np.interp(dist, along, raw_elev)
    return CourseProfile(
        points=points,
        dist_m=dist,
        elev_m=smooth_series(elev, smoothing_samples),
        spacing_m=spacing_m,
    )


def _mile_range(target: Target, total_mi: float) -> tuple[float, float]:
    """Parse the race-location text of a target into a mile range."""
    text = target.race_location
    match = _MILE_RANGE.search(text)
    if match:
        return float(match.group(1)), float(match.group(2))
    single = _SINGLE_MILE.search(text)
    if single:
        # "mile 3.0" names a point, so search a window around it that is wide
        # enough for the target's longest permitted segment to fit.
        centre = float(single.group(1))
        half = max(0.75, target.length_mi[1])
        return max(0.0, centre - half), min(total_mi, centre + half)
    return 0.0, total_mi


def compare(
    config: Config, profile: CourseProfile, only: str | None = None
) -> list[dict[str, object]]:
    """Measure the course at each target's location and compare to its band.

    Args:
        config: Loaded configuration.
        profile: The course profile.
        only: Restrict to a single target key.

    Returns:
        One dictionary per target with the measured values and a verdict.
    """
    rows: list[dict[str, object]] = []
    for target in config.targets:
        if only and target.key != only:
            continue
        start_mi, end_mi = _mile_range(target, profile.total_mi)
        want_steep = abs(target.ideal_grade_pct) >= 1.0
        grade, win_start, win_end, window_mi = profile.best_window(
            start_mi, end_mi, target.length_mi, steepest=want_steep
        )
        overall = profile.grade_between(start_mi, end_mi)
        inside = target.grade_matches(grade)
        rows.append(
            {
                "key": target.key,
                "label": target.label,
                "race_location": target.race_location,
                "searched_mi": (round(start_mi, 2), round(end_mi, 2)),
                "window_mi": (round(win_start, 2), round(win_end, 2)),
                "window_length_mi": round(window_mi, 2),
                "measured_grade_pct": round(grade, 2),
                "range_mean_grade_pct": round(overall, 2),
                "configured_band_pct": target.grade_pct,
                "ideal_grade_pct": target.ideal_grade_pct,
                "in_band": inside,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the course cross-check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gpx",
        nargs="?",
        default=None,
        help="course GPX; defaults to course_gpx in config.yaml",
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--feature", default=None, help="only this target key")
    args = parser.parse_args(argv)

    configure_logging()
    config = load_config(args.config)
    gpx = args.gpx or config.course_gpx
    if not gpx:
        parser.error("no GPX given and course_gpx is not set in config.yaml")

    analysis = config.analysis
    profile = load_course(
        Path(gpx),
        spacing_m=float(analysis.get("sample_spacing_m", 20.0)),
        smoothing_samples=int(analysis.get("smoothing_samples", 5)),
    )
    LOG.info(
        "course: %.2f mi, %.0f-%.0f m elevation",
        profile.total_mi,
        float(profile.elev_m.min()),
        float(profile.elev_m.max()),
    )

    rows = compare(config, profile, args.feature)
    print()
    print(
        f"{'feature':<16} {'race loc':<12} {'measured':>9} "
        f"{'configured band':>18} {'verdict':>9}"
    )
    print("-" * 70)
    for row in rows:
        band = row["configured_band_pct"]
        verdict = "in band" if row["in_band"] else "OUTSIDE"
        print(
            f"{row['key']:<16} {str(row['race_location'])[:12]:<12} "
            f"{row['measured_grade_pct']:>8.2f}% "
            f"{f'{band[0]:+.1f} to {band[1]:+.1f}%':>18} {verdict:>9}"
        )
    print()
    print(
        "Measured values are the steepest (or flattest, for the flat target) "
        "window inside the race-location range, swept across the target's "
        "whole length band and smoothed the same way the SLO road profiles "
        "are. Grade falls off sharply with window length, so the length at "
        "which a feature is most pronounced is the one reported."
    )

    outside = [row for row in rows if not row["in_band"]]
    if outside:
        print()
        print("Targets whose band does not contain the course measurement:")
        for row in outside:
            print(
                f"  - {row['label']}: course does "
                f"{row['measured_grade_pct']:+.2f}% over "
                f"{row['window_length_mi']} mi at miles "
                f"{row['window_mi'][0]}-{row['window_mi'][1]}, "
                f"band is {row['configured_band_pct'][0]:+.1f} to "
                f"{row['configured_band_pct'][1]:+.1f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
