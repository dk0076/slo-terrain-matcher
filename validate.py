"""Calibrate the DEM against barometric elevation from your own runs.

Takes a FIT file, pulls out the GPS track and the barometric altimeter trace,
samples the DEM at the same points, and reports how far apart they are. The
point is not the absolute offset -- a barometer drifts and the DEM has its own
datum -- but whether the *grades* agree, because grade is what the matcher
reports.

Usage::

    python validate.py run.fit
    python validate.py runs/*.fit --csv output/dem_validation.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import geo
from config import Config, configure_logging, load_config

LOG = logging.getLogger("validate")

#: Semicircle-to-degree conversion used by the FIT format.
SEMICIRCLE_TO_DEG = 180.0 / 2**31

#: Metres to feet.
FEET_PER_METRE = 3.280839895


@dataclass
class FitTrack:
    """A GPS track with barometric elevation, read from a FIT file."""

    path: Path
    coords: np.ndarray  # (n, 2) of (lon, lat)
    elev_m: np.ndarray  # barometric or GPS altitude
    dist_m: np.ndarray  # cumulative along-track distance
    altitude_source: str

    @property
    def length_mi(self) -> float:
        """Track length in miles."""
        return geo.metres_to_miles(float(self.dist_m[-1]))


@dataclass
class ValidationReport:
    """How the DEM compares to one recorded run."""

    track: FitTrack
    sample_count: int
    coverage: float
    mean_offset_m: float
    median_offset_m: float
    offset_stdev_m: float
    elevation_rmse_m: float
    detrended_rmse_m: float
    grade_window_m: float
    grade_bias_pct: float
    grade_mae_pct: float
    grade_rmse_pct: float
    grade_correlation: float
    fit_gain_m: float
    dem_gain_m: float

    def summary(self) -> str:
        """A readable multi-line summary of this report."""
        return "\n".join(
            [
                f"{self.track.path.name}: {self.track.length_mi:.2f} mi, "
                f"{self.sample_count} samples "
                f"({self.coverage * 100:.0f}% inside the DEM), "
                f"altitude from {self.track.altitude_source}",
                "",
                "  Elevation",
                f"    mean offset (FIT - DEM) : {self.mean_offset_m:+7.2f} m "
                f"({self.mean_offset_m * FEET_PER_METRE:+.0f} ft)",
                f"    median offset           : {self.median_offset_m:+7.2f} m",
                f"    offset std dev          : {self.offset_stdev_m:7.2f} m",
                f"    RMSE                    : {self.elevation_rmse_m:7.2f} m",
                f"    RMSE after removing the constant offset"
                f" : {self.detrended_rmse_m:.2f} m",
                "",
                f"  Grade over {self.grade_window_m:.0f} m windows",
                f"    bias (FIT - DEM)        : {self.grade_bias_pct:+7.3f} %",
                f"    mean absolute error     : {self.grade_mae_pct:7.3f} %",
                f"    RMSE                    : {self.grade_rmse_pct:7.3f} %",
                f"    correlation             : {self.grade_correlation:7.3f}",
                "",
                "  Climbing",
                f"    FIT total gain          : {self.fit_gain_m:7.1f} m "
                f"({self.fit_gain_m * FEET_PER_METRE:.0f} ft)",
                f"    DEM total gain          : {self.dem_gain_m:7.1f} m "
                f"({self.dem_gain_m * FEET_PER_METRE:.0f} ft)",
            ]
        )

    def verdict(self) -> str:
        """A one-line judgement on whether the DEM grades can be trusted."""
        if self.grade_mae_pct <= 0.5 and self.grade_correlation >= 0.9:
            return (
                "DEM grades track the barometer closely; the reported grades "
                "are trustworthy at this window length."
            )
        if self.grade_mae_pct <= 1.0 and self.grade_correlation >= 0.75:
            return (
                "DEM grades are usable but loose: expect roughly "
                f"{self.grade_mae_pct:.1f} percentage points of error on any "
                "individual segment."
            )
        return (
            "DEM grades disagree with the barometer here. Treat the matcher's "
            "grades as indicative only, and check whether the run included "
            "bridges, tree cover or tunnels, where a bare-earth DEM and a "
            "barometer cannot agree."
        )


def read_fit(path: Path) -> FitTrack:
    """Extract the GPS track and altitude series from a FIT file.

    Prefers ``enhanced_altitude`` and then ``altitude``, which on a watch with
    a barometer are the barometric trace. Falls back to GPS altitude only if
    nothing else is present, and says so in ``altitude_source``.

    Args:
        path: Path to the ``.fit`` file.

    Returns:
        The parsed track.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file holds no positioned records with altitude.
    """
    import fitdecode

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FIT file not found: {path}")

    lats: list[float] = []
    lons: list[float] = []
    elevations: list[float] = []
    source_fields: set[str] = set()

    with fitdecode.FitReader(str(path)) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name != "record":
                continue

            lat = _field(frame, "position_lat")
            lon = _field(frame, "position_long")
            if lat is None or lon is None:
                continue

            altitude = None
            for field_name in ("enhanced_altitude", "altitude"):
                value = _field(frame, field_name)
                if value is not None:
                    altitude = value
                    source_fields.add(field_name)
                    break
            if altitude is None:
                continue

            # FIT stores lat/lon in semicircles unless the SDK already scaled
            # them; values beyond the degree range mean raw semicircles.
            if abs(lat) > 180.0:
                lat = lat * SEMICIRCLE_TO_DEG
            if abs(lon) > 180.0:
                lon = lon * SEMICIRCLE_TO_DEG

            lats.append(float(lat))
            lons.append(float(lon))
            elevations.append(float(altitude))

    if len(lats) < 10:
        raise ValueError(
            f"{path.name} has only {len(lats)} positioned records with "
            "altitude; need at least 10 to compare anything"
        )

    coords = np.column_stack([np.array(lons), np.array(lats)])
    elev = np.array(elevations, dtype=float)
    dist = geo.cumulative_distance_m(coords)

    source = (
        "enhanced_altitude (barometric)"
        if "enhanced_altitude" in source_fields
        else "altitude"
    )
    LOG.info(
        "%s: %d records, %.2f mi, altitude field %s",
        path.name,
        len(lats),
        geo.metres_to_miles(float(dist[-1])),
        source,
    )
    return FitTrack(
        path=path,
        coords=coords,
        elev_m=elev,
        dist_m=dist,
        altitude_source=source,
    )


def _field(frame, name: str):
    """Read one FIT field, returning ``None`` when absent or unset."""
    try:
        if not frame.has_field(name):
            return None
        value = frame.get_value(name)
    except (KeyError, ValueError):  # pragma: no cover - malformed frames
        return None
    return None if value is None else value


def _grades(
    dist: np.ndarray, elev: np.ndarray, window_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grades over non-overlapping windows of roughly ``window_m``.

    Returns:
        ``(window_midpoint_distances, grades_pct)``.
    """
    edges = [0]
    cursor = 0
    while True:
        target = dist[cursor] + window_m
        nxt = int(np.searchsorted(dist, target))
        if nxt >= dist.size:
            break
        if nxt <= cursor:
            nxt = cursor + 1
        edges.append(nxt)
        cursor = nxt

    if len(edges) < 2:
        return np.empty(0), np.empty(0)

    starts = np.array(edges[:-1])
    ends = np.array(edges[1:])
    run = dist[ends] - dist[starts]
    valid = run > 0
    grades = np.zeros(starts.size, dtype=float)
    grades[valid] = (
        (elev[ends][valid] - elev[starts][valid]) / run[valid] * 100.0
    )
    midpoints = (dist[starts] + dist[ends]) / 2.0
    return midpoints[valid], grades[valid]


def validate_track(
    track: FitTrack, dem_path: Path, config: Config
) -> ValidationReport:
    """Compare one recorded track against the DEM.

    Args:
        track: The parsed FIT track.
        dem_path: Path to the clipped DEM GeoTIFF.
        config: Loaded configuration, for the grade window and smoothing.

    Returns:
        The comparison report.

    Raises:
        ValueError: If too little of the track falls inside the DEM.
    """
    from analyze import DemSampler, smooth_series

    sampler = DemSampler(dem_path)
    dem_elev = sampler.sample(track.coords[:, 0], track.coords[:, 1])

    inside = np.isfinite(dem_elev)
    coverage = float(inside.mean())
    if coverage < 0.5:
        raise ValueError(
            f"only {coverage * 100:.0f}% of {track.path.name} falls inside the "
            "DEM; widen area.bbox in config.yaml to cover this run"
        )

    dist = track.dist_m[inside]
    fit_elev = track.elev_m[inside]
    dem_values = dem_elev[inside]

    # Smooth both series identically, so the comparison is between what the
    # matcher actually uses and what the watch actually recorded, rather than
    # between one smoothed and one raw signal.
    smoothing = int(config.analysis.get("smoothing_samples", 5))
    fit_smooth = smooth_series(fit_elev, smoothing)
    dem_smooth = smooth_series(dem_values, smoothing)

    offsets = fit_smooth - dem_smooth
    mean_offset = float(np.mean(offsets))
    rmse = float(np.sqrt(np.mean(offsets**2)))
    detrended = float(np.sqrt(np.mean((offsets - mean_offset) ** 2)))

    window_m = float(config.analysis.get("min_window_m", 100.0))
    _, fit_grades = _grades(dist, fit_smooth, window_m)
    _, dem_grades = _grades(dist, dem_smooth, window_m)
    count = min(fit_grades.size, dem_grades.size)
    fit_grades, dem_grades = fit_grades[:count], dem_grades[:count]

    if count >= 2:
        differences = fit_grades - dem_grades
        bias = float(np.mean(differences))
        mae = float(np.mean(np.abs(differences)))
        grade_rmse = float(np.sqrt(np.mean(differences**2)))
        if np.std(fit_grades) > 1e-9 and np.std(dem_grades) > 1e-9:
            correlation = float(np.corrcoef(fit_grades, dem_grades)[0, 1])
        else:
            correlation = float("nan")
    else:
        bias = mae = grade_rmse = correlation = float("nan")

    fit_diff = np.diff(fit_smooth)
    dem_diff = np.diff(dem_smooth)

    return ValidationReport(
        track=track,
        sample_count=int(inside.sum()),
        coverage=coverage,
        mean_offset_m=mean_offset,
        median_offset_m=float(np.median(offsets)),
        offset_stdev_m=float(np.std(offsets)),
        elevation_rmse_m=rmse,
        detrended_rmse_m=detrended,
        grade_window_m=window_m,
        grade_bias_pct=bias,
        grade_mae_pct=mae,
        grade_rmse_pct=grade_rmse,
        grade_correlation=correlation,
        fit_gain_m=float(fit_diff[fit_diff > 0].sum()),
        dem_gain_m=float(dem_diff[dem_diff > 0].sum()),
    )


def write_csv(reports: Sequence[ValidationReport], path: Path) -> Path:
    """Write one row per validated track.

    Args:
        reports: Reports to write.
        path: Destination CSV.

    Returns:
        The path written.
    """
    import csv

    columns = [
        "file",
        "length_mi",
        "samples",
        "coverage_pct",
        "altitude_source",
        "mean_offset_m",
        "median_offset_m",
        "offset_stdev_m",
        "elevation_rmse_m",
        "detrended_rmse_m",
        "grade_window_m",
        "grade_bias_pct",
        "grade_mae_pct",
        "grade_rmse_pct",
        "grade_correlation",
        "fit_gain_m",
        "dem_gain_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for report in reports:
            writer.writerow(
                [
                    report.track.path.name,
                    round(report.track.length_mi, 3),
                    report.sample_count,
                    round(report.coverage * 100, 1),
                    report.track.altitude_source,
                    round(report.mean_offset_m, 2),
                    round(report.median_offset_m, 2),
                    round(report.offset_stdev_m, 2),
                    round(report.elevation_rmse_m, 2),
                    round(report.detrended_rmse_m, 2),
                    round(report.grade_window_m, 1),
                    round(report.grade_bias_pct, 4),
                    round(report.grade_mae_pct, 4),
                    round(report.grade_rmse_pct, 4),
                    round(report.grade_correlation, 4),
                    round(report.fit_gain_m, 1),
                    round(report.dem_gain_m, 1),
                ]
            )
    LOG.info("wrote %s (%d rows)", path.name, len(reports))
    return path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for DEM validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", nargs="+", help="one or more .fit files")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--csv", default=None, help="also write a summary CSV to this path"
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = load_config(args.config)

    from fetch_data import fetch_dem

    dem_path = fetch_dem(config)

    paths: list[Path] = []
    for pattern in args.fit:
        candidate = Path(pattern)
        if candidate.exists():
            paths.append(candidate)
        else:
            matches = sorted(Path().glob(pattern))
            if not matches:
                LOG.error("no file matches %s", pattern)
            paths.extend(matches)

    if not paths:
        LOG.error("nothing to validate")
        return 1

    reports: list[ValidationReport] = []
    for path in paths:
        try:
            track = read_fit(path)
            report = validate_track(track, dem_path, config)
        except (ValueError, FileNotFoundError) as error:
            LOG.error("%s: %s", path.name, error)
            continue
        reports.append(report)
        print()
        print(report.summary())
        print()
        print(f"  Verdict: {report.verdict()}")

    if not reports:
        return 1

    if len(reports) > 1:
        maes = np.array([report.grade_mae_pct for report in reports])
        offsets = np.array([report.mean_offset_m for report in reports])
        print()
        print("=" * 68)
        print(
            f"Across {len(reports)} runs: mean grade error "
            f"{np.nanmean(maes):.3f} %, offset spread "
            f"{np.nanstd(offsets):.2f} m"
        )

    if args.csv:
        write_csv(reports, Path(args.csv))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
