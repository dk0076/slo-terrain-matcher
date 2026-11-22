"""Replicate any race course on local roads.

Give it a course GPX and somewhere to run, and it builds the closest route it
can find in that area::

    python replicate.py --course oakland.gpx --area "San Luis Obispo, CA"

The course can be any race; the area can be any place name, a centre point, or
a raw bounding box. Nothing needs editing in ``config.yaml``.

By default the replica is the same length as the course. Ask for a shorter one
and it reproduces the course's features in the same order and proportion, at
their true grades, over a distance you can actually run::

    python replicate.py --course oakland.gpx --area "Boulder, CO" --miles 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import geo
from area import Area, looks_like_united_states, resolve_area
from config import Config, configure_logging, load_config

LOG = logging.getLogger("replicate")

#: Metres to feet.
FEET_PER_METRE = 3.280839895


def replicate(
    config: Config,
    area: Area,
    course_gpx: Path,
    miles: float | None,
    out_path: Path,
    *,
    beam: int = 600,
    emphasis: float = 0.6,
    per_node: int = 3,
    unpaved_penalty: float = 0.9,
    drift_weight: float = 1.0,
    recent_memory: int | None = None,
    point_to_point: bool = False,
    start: tuple[float, float] | None = None,
    refresh: bool = False,
    on_progress=None,
    route_name: str | None = None,
):
    """Build the closest replica of a course within an area.

    Args:
        config: Loaded configuration, already pointed at the area.
        area: The resolved search area.
        course_gpx: Path to the race course GPX.
        miles: Replica length; ``None`` means match the course's own length.
        out_path: Where to write the replica GPX.
        beam: Beam width for the route search.
        emphasis: How hard to chase the course's steep sections.
        per_node: Beam diversity cap.
        unpaved_penalty: Extra cost per metre of known-unpaved surface.
        drift_weight: How hard to hold the replica to the course's own
            elevation curve. 0 disables the correction.
        recent_memory: How many recent edges the search refuses to reuse.
            ``None`` scales it with the distance: a long route inside a small
            area has to lap, so it needs a shorter memory than a short one.
        point_to_point: Do not close the route back to the start.
        start: Optional ``(latitude, longitude)`` start point.
        refresh: Ignore cached downloads.
        on_progress: Optional ``callable(percent, stage)`` for a caller
            showing progress.
        route_name: Override the name given to the finished route.

    Returns:
        ``(route, chart_path)``. The route is ``None`` if none could be
        built; the chart path is ``None`` if the chart could not be drawn.

    Raises:
        FileNotFoundError: If the course GPX does not exist.
    """
    import analyze
    import course as course_module
    import fetch_data
    import route as route_module

    course_gpx = Path(course_gpx)
    if not course_gpx.exists():
        raise FileNotFoundError(f"course GPX not found: {course_gpx}")

    spacing = float(config.analysis.get("sample_spacing_m", 20.0))
    smoothing = int(config.analysis.get("smoothing_samples", 5))

    def report(percent: int, stage: str) -> None:
        if on_progress is not None:
            on_progress(percent, stage)

    report(5, "reading the course")
    LOG.info("--- 1/4: reading the course ---")
    profile = course_module.load_course(course_gpx, spacing, smoothing)
    course_miles = profile.total_mi
    target_miles = miles if miles else course_miles
    LOG.info(
        "course %s: %.2f mi; replica target %.2f mi",
        course_gpx.name,
        course_miles,
        target_miles,
    )

    report(12, "fetching elevation data")
    LOG.info("--- 2/4: fetching elevation and roads for %s ---", area.label)
    dem_path = fetch_data.fetch_dem(config, refresh=refresh)
    report(22, "fetching roads and paths (the slow part)")
    osm_payload = fetch_data.fetch_osm(config, refresh=refresh)

    report(48, "building the road graph")
    LOG.info("--- 3/4: building the road graph ---")
    edges, _ = analyze.build_edges(osm_payload)
    sampler = analyze.DemSampler(dem_path)
    target = route_module.course_target(profile)
    graph = route_module.RoadGraph(edges, sampler, spacing, smoothing)

    start_lat, start_lon = start if start else area.center
    start_node = graph.nearest_node(start_lat, start_lon)

    if recent_memory is None:
        # 250 suits a route of a dozen miles or so. A longer route in the same
        # area must reuse roads, so the memory shortens in proportion rather
        # than making the search impossible.
        recent_memory = int(max(60, min(250, 250 * (12.0 / max(1.0, target_miles)))))
        LOG.info("route memory: %d edges", recent_memory)

    report(60, "searching for the closest route")
    LOG.info("--- 4/4: searching for the closest replica ---")
    name = route_name or f"{course_gpx.stem} replica — {target_miles:g} mi"

    def search_progress(fraction: float) -> None:
        report(60 + int(33 * max(0.0, min(1.0, fraction))), "searching for the closest route")
    built = route_module.build_route(
        graph,
        target,
        geo.miles_to_metres(target_miles),
        start_node,
        name=name,
        beam_width=beam,
        recent_memory=recent_memory,
        unpaved_penalty=unpaved_penalty,
        feature_emphasis=emphasis,
        per_node=per_node,
        drift_weight=drift_weight,
        close_loop=not point_to_point,
        length_tolerance=(0.97, 1.03),
        on_progress=search_progress,
    )
    if built is None:
        return None, None

    report(95, "writing the route and chart")
    route_module.write_gpx(built, out_path)

    from chart import write_profile_comparison

    chart_path = out_path.with_suffix(".html")
    try:
        write_profile_comparison(
            built, profile, chart_path, course_name=course_gpx.name
        )
    except ValueError as error:
        LOG.warning("could not chart the comparison: %s", error)
        chart_path = None
    return built, chart_path


def print_summary(built, area: Area, course_name: str, out_path: Path) -> None:
    """Print a readable comparison of the replica against the course."""
    fit = built.fit
    gain_ft = built.gain_m * FEET_PER_METRE
    course_gain_ft = fit.get("course_equivalent_gain_ft", 0.0)

    print()
    print(f"{built.name}")
    print(f"  area              {area.label}")
    print(f"  length            {built.length_mi:.2f} mi")
    print(
        f"  climbing          {gain_ft:.0f} ft  "
        f"(the course over the same distance: {course_gain_ft:.0f} ft)"
    )
    if course_gain_ft > 0:
        ratio = gain_ft / course_gain_ft
        verdict = (
            "close" if 0.85 <= ratio <= 1.15
            else ("hillier than the course" if ratio > 1 else "flatter than the course")
        )
        print(f"                    {ratio:.2f}x — {verdict}")
    print(
        f"  steepest climb    {fit.get('steepest_climb_pct', 0):+.1f}%  "
        f"(course {fit.get('course_steepest_climb_pct', 0):+.1f}%)"
    )
    print(
        f"  steepest descent  {fit.get('steepest_descent_pct', 0):+.1f}%  "
        f"(course {fit.get('course_steepest_descent_pct', 0):+.1f}%)"
    )
    print(
        f"  grade error       {fit.get('grade_mae_pct', 0):.2f} percentage points"
    )
    print(
        f"  elevation drift   {fit.get('mean_drift_ft', 0):.0f} ft average, "
        f"{fit.get('max_drift_ft', 0):.0f} ft worst  "
        "(how far from the course's own height)"
    )

    correlation = fit.get("grade_correlation", float("nan"))
    if correlation >= 0.7:
        quality = "tracks the course closely"
    elif correlation >= 0.4:
        quality = "follows the course's shape, loosely"
    elif correlation >= 0.15:
        quality = "weak; the terrain here does not offer much of a match"
    else:
        quality = "poor; consider a different area or distance"
    print(f"  profile match     {correlation:.2f} correlation — {quality}")

    paved = built.surface_fractions.get("paved", 0.0) * 100
    unpaved = built.surface_fractions.get("unpaved", 0.0) * 100
    print(f"  surface           {paved:.0f}% known paved, {unpaved:.0f}% known unpaved")
    print(f"  repeated ground   {built.repeat_fraction * 100:.0f}% of the distance")
    print(f"  closed loop       {'yes' if built.closed_loop else 'no'}")
    if built.names:
        print(f"  streets           {', '.join(built.names[:12])}")
    print()
    print(f"  written to {out_path}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Build the closest local replica of a race course.",
        epilog=(
            "Examples:\n"
            '  python replicate.py --course oakland.gpx --area "San Luis Obispo, CA"\n'
            '  python replicate.py --course boston.gpx --center 35.28,-120.66 --miles 12\n'
            "  python replicate.py --course race.gpx --bbox -120.75,35.15,-120.55,35.35"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--course", required=True, help="the race course GPX to replicate"
    )

    where = parser.add_argument_group("where to run (give one)")
    where.add_argument("--area", default=None, help='a place name, e.g. "Boulder, CO"')
    where.add_argument("--center", default=None, help="'lat,lon'")
    where.add_argument("--bbox", default=None, help="'west,south,east,north'")
    where.add_argument(
        "--radius",
        type=float,
        default=8.0,
        help="miles around the place or centre point (default 8)",
    )

    shape = parser.add_argument_group("the replica")
    shape.add_argument(
        "--miles",
        type=float,
        default=None,
        help="replica length; defaults to the course's own length",
    )
    shape.add_argument(
        "--start", default=None, help="'lat,lon' to start from; defaults to the area centre"
    )
    shape.add_argument(
        "--point-to-point",
        action="store_true",
        help="do not close the route back to the start",
    )

    tuning = parser.add_argument_group("tuning")
    tuning.add_argument("--beam", type=int, default=600, help="search width")
    tuning.add_argument(
        "--emphasis",
        type=float,
        default=0.6,
        help="how hard to chase the course's steep sections",
    )
    tuning.add_argument("--per-node", type=int, default=3, help="beam diversity cap")
    tuning.add_argument(
        "--unpaved-penalty", type=float, default=0.9, help="cost per metre of dirt"
    )
    tuning.add_argument(
        "--memory",
        type=int,
        default=None,
        help=(
            "edges the search refuses to reuse; higher means less doubling "
            "back. Defaults to scaling with the distance"
        ),
    )
    tuning.add_argument(
        "--drift",
        type=float,
        default=1.0,
        help=(
            "how hard to hold the replica to the course's elevation curve; "
            "0 disables it"
        ),
    )

    parser.add_argument("--out", default=None, help="output .gpx path")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    started = time.time()

    try:
        area = resolve_area(
            place=args.area,
            center=args.center,
            bbox=args.bbox,
            radius_mi=args.radius,
        )
    except (ValueError, LookupError) as error:
        LOG.error("%s", error)
        return 2

    if not looks_like_united_states(area):
        LOG.error(
            "that area looks to be outside the United States. Elevation comes "
            "from the USGS 3D Elevation Program, which is US-only, so there "
            "is no terrain data to work from there."
        )
        return 2

    config = load_config(args.config).with_area(
        area.bbox, center=area.center, name=area.label
    )

    start = None
    if args.start:
        try:
            lat_text, lon_text = args.start.split(",")
            start = (float(lat_text), float(lon_text))
        except ValueError:
            LOG.error("--start needs 'lat,lon', got %r", args.start)
            return 2

    course_path = Path(args.course)
    out_path = (
        Path(args.out)
        if args.out
        else config.output_dir / f"{course_path.stem}-replica.gpx"
    )

    try:
        built, chart_path = replicate(
            config,
            area,
            course_path,
            args.miles,
            out_path,
            beam=args.beam,
            emphasis=args.emphasis,
            per_node=args.per_node,
            unpaved_penalty=args.unpaved_penalty,
            drift_weight=args.drift,
            recent_memory=args.memory,
            point_to_point=args.point_to_point,
            start=start,
            refresh=args.refresh,
        )
    except FileNotFoundError as error:
        LOG.error("%s", error)
        return 2
    except ValueError as error:
        LOG.error("%s", error)
        return 1

    if built is None:
        LOG.error(
            "could not build a route of that length here. The road network "
            "around the start may be too sparse to close a loop; try a larger "
            "--radius, a different --start, or a shorter --miles."
        )
        return 1

    print_summary(built, area, course_path.name, out_path)
    if chart_path is not None:
        print(f"  chart      {chart_path}")
    LOG.info("done in %.1fs", time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
