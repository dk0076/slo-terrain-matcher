"""Run the whole pipeline: fetch, analyse, report.

Usage::

    python run.py                  # fetch (cached), analyse, write all outputs
    python run.py --refresh        # ignore the download cache
    python run.py --skip-waterways # skip the extra Overpass call
    python run.py --check-course   # also re-measure targets against the GPX
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from analyze import DemSampler, run_analysis
from config import configure_logging, load_config
from fetch_data import fetch_dem, fetch_osm, fetch_waterways
from report import write_all
from route import build_routes, write_gpx

LOG = logging.getLogger("run")


def main(argv: list[str] | None = None) -> int:
    """Execute every stage of the pipeline in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached downloads"
    )
    parser.add_argument(
        "--skip-waterways",
        action="store_true",
        help="skip the waterway fetch; creek crossings will not be flagged",
    )
    parser.add_argument(
        "--check-course",
        action="store_true",
        help="re-measure the configured targets against course_gpx first",
    )
    parser.add_argument(
        "--skip-routes",
        action="store_true",
        help="skip building the whole routes configured under `routes`",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="debug-level logging"
    )
    args = parser.parse_args(argv)

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    started = time.time()
    config = load_config(args.config)
    LOG.info(
        "searching %s for %d course features",
        config.area.get("name", "the configured bbox"),
        len(config.targets),
    )

    if args.check_course:
        if not config.course_gpx:
            LOG.warning("--check-course given but course_gpx is not set")
        else:
            from course import compare, load_course

            profile = load_course(
                Path(config.course_gpx),
                spacing_m=float(config.analysis.get("sample_spacing_m", 20.0)),
                smoothing_samples=int(
                    config.analysis.get("smoothing_samples", 5)
                ),
            )
            for row in compare(config, profile):
                LOG.info(
                    "course check %-16s measured %+6.2f%%  band %+.1f to %+.1f%%  %s",
                    row["key"],
                    row["measured_grade_pct"],
                    row["configured_band_pct"][0],
                    row["configured_band_pct"][1],
                    "in band" if row["in_band"] else "OUTSIDE BAND",
                )

    LOG.info("--- stage 1/3: data acquisition ---")
    dem_path = fetch_dem(config, refresh=args.refresh)
    osm_payload = fetch_osm(config, refresh=args.refresh)
    waterways = (
        None
        if args.skip_waterways
        else fetch_waterways(config, refresh=args.refresh)
    )

    LOG.info("--- stage 2/3: grade analysis ---")
    result = run_analysis(config, dem_path, osm_payload, waterways)

    LOG.info("--- stage 3/4: reports ---")
    paths = write_all(result)

    LOG.info("--- stage 4/4: whole routes ---")
    route_paths = []
    if args.skip_routes:
        LOG.info("skipped (--skip-routes)")
    elif not config.routes:
        LOG.info("no routes configured; nothing to build")
    elif not config.course_gpx:
        LOG.warning("routes are configured but course_gpx is not set; skipping")
    else:
        from course import load_course

        profile = load_course(
            Path(config.course_gpx),
            spacing_m=float(config.analysis.get("sample_spacing_m", 20.0)),
            smoothing_samples=int(config.analysis.get("smoothing_samples", 5)),
        )
        # Reuse the edges the analysis already built; only the DEM sampler
        # has to be reopened.
        sampler = DemSampler(dem_path)
        for built in build_routes(config, result.edges, sampler, profile):
            slug = built.name.lower().replace(" ", "-")
            path = config.output_dir / f"route-{slug}.gpx"
            write_gpx(built, path)
            route_paths.append((built, path))
            try:
                from chart import write_profile_comparison

                write_profile_comparison(
                    built,
                    profile,
                    path.with_suffix(".html"),
                    course_name=Path(config.course_gpx).name,
                )
            except ValueError as error:
                LOG.warning("could not chart %s: %s", built.name, error)

    elapsed = time.time() - started
    total = sum(len(matches) for matches in result.matches.values())
    LOG.info("done in %.1fs: %d matching segments", elapsed, total)
    print()
    print("Outputs:")
    for label, path in (
        ("all matches (CSV)", paths.csv),
        ("report (Markdown)", paths.markdown),
        ("top segments (GPX)", paths.gpx),
        ("map (HTML)", paths.map),
    ):
        print(f"  {label:<20} {path}")
    for built, path in route_paths:
        print(f"  {built.name:<20} {path}")
    if route_paths:
        print()
        print("Whole routes:")
        for built, _ in route_paths:
            fit = built.fit
            print(
                f"  {built.name}: {built.length_mi:.2f} mi, "
                f"{built.gain_m * 3.280839895:.0f} ft gain, "
                f"grade error {fit.get('grade_mae_pct', 0):.2f} pts, "
                f"match {fit.get('grade_correlation', 0):.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
