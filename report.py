"""Ranking and output: CSV, Markdown, GPX and an interactive map.

Writes four files into the configured output directory:

``matches.csv``
    Every match, one row per segment, machine-readable.
``report.md``
    The top matches per feature with the caveats that apply to them.
``routes.gpx``
    The best few segments per feature as GPX tracks for a watch.
``map.html``
    A Folium map of the top matches, coloured by feature.
"""

from __future__ import annotations

import csv
import html
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import folium

import geo
from analyze import AnalysisResult, Match
from config import Config, Target

LOG = logging.getLogger("report")

#: Column order for ``matches.csv``.
CSV_COLUMNS = [
    "target_key",
    "target_label",
    "rank",
    "score",
    "name",
    "highway_classes",
    "length_mi",
    "length_m",
    "mean_grade_pct",
    "steepest_grade_pct",
    "max_grade_pct",
    "min_grade_pct",
    "gain_m",
    "gain_ft",
    "loss_m",
    "surface",
    "paved_fraction",
    "loopable",
    "loop_length_mi",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "distance_from_reference_mi",
    "dem_reliable",
    "flags",
    "google_maps_link",
    "corridor_id",
]

#: Metres to feet.
FEET_PER_METRE = 3.280839895


@dataclass(frozen=True)
class ReportPaths:
    """Where each generated artefact landed."""

    csv: Path
    markdown: Path
    gpx: Path
    map: Path


def _row(match: Match, target: Target, rank: int) -> dict[str, object]:
    """Flatten one match into a CSV row."""
    loop_mi = (
        round(geo.metres_to_miles(match.loop_length_m), 3)
        if match.loop_length_m is not None
        else ""
    )
    loopable = "" if match.loopable is None else str(bool(match.loopable)).lower()
    return {
        "target_key": target.key,
        "target_label": target.label,
        "rank": rank,
        "score": match.score,
        "name": match.name,
        "highway_classes": match.highways,
        "length_mi": round(match.length_mi, 3),
        "length_m": round(match.length_m, 1),
        "mean_grade_pct": round(match.mean_grade_pct, 2),
        "steepest_grade_pct": round(match.steepest_grade_pct, 2),
        "max_grade_pct": round(match.max_grade_pct, 2),
        "min_grade_pct": round(match.min_grade_pct, 2),
        "gain_m": round(match.gain_m, 1),
        "gain_ft": round(match.gain_m * FEET_PER_METRE, 0),
        "loss_m": round(match.loss_m, 1),
        "surface": match.surface,
        "paved_fraction": round(match.paved_fraction, 2),
        "loopable": loopable,
        "loop_length_mi": loop_mi,
        "start_lat": round(match.start_lat, 6),
        "start_lon": round(match.start_lon, 6),
        "end_lat": round(match.end_lat, 6),
        "end_lon": round(match.end_lon, 6),
        "distance_from_reference_mi": round(match.distance_from_reference_mi, 2),
        "dem_reliable": str(not match.unreliable).lower(),
        "flags": "; ".join(match.unreliable_reasons),
        "google_maps_link": geo.google_maps_link(match.start_lat, match.start_lon),
        "corridor_id": match.corridor_id,
    }


def write_csv(result: AnalysisResult, path: Path) -> Path:
    """Write every match to CSV, ranked within each target.

    Args:
        result: The analysis result.
        path: Destination file.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        total = 0
        for target in result.config.targets:
            for rank, match in enumerate(result.matches.get(target.key, []), 1):
                writer.writerow(_row(match, target, rank))
                total += 1
    LOG.info("wrote %s (%d rows)", path.name, total)
    return path


def _grade_text(match: Match) -> str:
    """Mean and steepest grade as one readable cell."""
    return f"{match.mean_grade_pct:+.2f}% (max {match.steepest_grade_pct:+.1f}%)"


def _loop_text(match: Match, target: Target) -> str:
    """Loopability as a readable cell."""
    if not target.want_loopable:
        return "–"
    if match.loopable is None:
        return "not checked"
    if not match.loopable:
        return "out-and-back"
    if match.loop_length_m is None:
        return "yes"
    return f"yes, {geo.metres_to_miles(match.loop_length_m):.1f} mi loop"


def write_markdown(result: AnalysisResult, path: Path) -> Path:
    """Write the human-readable report.

    Args:
        result: The analysis result.
        path: Destination file.

    Returns:
        The path written.
    """
    config = result.config
    top_n = int(config.output.get("top_n_report", 10))
    ref_lat, ref_lon = config.reference_point
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# Terrain matches in {config.area.get('name', 'the search area')}")
    lines.append("")
    lines.append(
        f"Generated {now}. Distances are measured from the reference point "
        f"({ref_lat:.4f}, {ref_lon:.4f}) in a straight line, not by road."
    )
    lines.append("")

    # -- how to read it ---------------------------------------------------
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        f"Elevations come from the USGS 3DEP 1/3 arc-second DEM, whose cells are "
        f"about {result.dem_cell_m:.0f} m across in this area. That resolution is "
        "dependable for a grade averaged over 50 m or more, but it cannot see a "
        "short pitch: a 30 m ramp between two cells is simply absent from the "
        f"data. Every window here is at least "
        f"{float(config.analysis.get('min_window_m', 100)):.0f} m long for that "
        "reason, and the elevation series is smoothed over "
        f"{int(config.analysis.get('smoothing_samples', 5))} samples "
        f"({int(config.analysis.get('smoothing_samples', 5)) * float(config.analysis.get('sample_spacing_m', 20)):.0f} m) "
        "before any grade is computed, so single-cell noise does not turn into a "
        "fake 15% wall."
    )
    lines.append("")
    lines.append(
        "**Max grade** is the steepest grade *sustained over "
        f"{float(config.analysis.get('min_window_m', 100)):.0f} m* inside the "
        "segment, in the direction the segment runs. It is not a point value."
    )
    lines.append("")
    lines.append(
        "**Surface** comes from the OSM `surface` tag. The tag is incomplete, so "
        "`unknown` means nobody has recorded it, **not** that it is paved. Many "
        "of the rural roads below are in fact sealed; several of the trails are "
        "not. Check anything you plan to run repeatedly."
    )
    lines.append("")
    lines.append(
        "**Flagged segments** carry a caveat in the notes column: the DEM is "
        "bare-earth, so a bridge or overpass reads as the ground beneath it, and "
        "a creek crossing can read as a notch. Those segments are kept but "
        "scored down."
    )
    lines.append("")

    # -- summary ----------------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    lines.append("| Feature | Race location | Spec | Candidates | Best score |")
    lines.append("|---|---|---|---|---|")
    for target in config.targets:
        matches = result.matches.get(target.key, [])
        best = f"{matches[0].score:.1f}" if matches else "—"
        lines.append(
            f"| {target.label} | {target.race_location} | {target.spec} | "
            f"{len(matches)} | {best} |"
        )
    lines.append("")

    # -- per target -------------------------------------------------------
    for target in config.targets:
        matches = result.matches.get(target.key, [])
        lines.append(f"## {target.label}")
        lines.append("")
        lines.append(
            f"*Race location:* {target.race_location} — *target:* {target.spec}"
        )
        lines.append("")
        if not matches:
            lines.append(
                "No segment in the search area matched this specification. "
                "Widening the grade or length band in `config.yaml`, or enlarging "
                "the bounding box, is the way to loosen it."
            )
            lines.append("")
            continue

        lines.append(
            "| # | Street | Length | Grade | Gain | Surface | Loop | From ref | "
            "Score | Start | Notes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for rank, match in enumerate(matches[:top_n], 1):
            notes = "; ".join(match.unreliable_reasons) or ""
            link = geo.google_maps_link(match.start_lat, match.start_lon)
            lines.append(
                f"| {rank} "
                f"| {match.name} "
                f"| {match.length_mi:.2f} mi "
                f"| {_grade_text(match)} "
                f"| {match.gain_m * FEET_PER_METRE:.0f} ft "
                f"| {match.surface} "
                f"| {_loop_text(match, target)} "
                f"| {match.distance_from_reference_mi:.1f} mi "
                f"| {match.score:.1f} "
                f"| [{match.start_lat:.4f}, {match.start_lon:.4f}]({link}) "
                f"| {notes} |"
            )
        lines.append("")
        if len(matches) > top_n:
            lines.append(
                f"{len(matches) - top_n} further candidates are in `matches.csv`."
            )
            lines.append("")

    # -- scope ------------------------------------------------------------
    lines.append("## Scope of the search")
    lines.append("")
    west, south, east, north = config.bbox
    lines.append(
        f"- Bounding box: {west}, {south} to {east}, {north}\n"
        f"- Network: {len(result.edges):,} road and path edges, "
        f"{sum(edge.length_m for edge in result.edges) / 1000:,.0f} km total\n"
        f"- Chained into {len(result.corridors):,} continuous corridors, "
        f"{len(result.profiles):,} of which were long enough to profile\n"
        f"- Sample spacing: "
        f"{float(config.analysis.get('sample_spacing_m', 20)):.0f} m"
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote %s", path.name)
    return path


def write_gpx(result: AnalysisResult, path: Path) -> Path:
    """Write the best segments per feature as GPX tracks.

    One track per segment, named so it is identifiable on a watch. Elevation
    is included from the DEM, which is useful on the device but is the same
    smoothed data the analysis used, not a survey.

    Args:
        result: The analysis result.
        path: Destination file.

    Returns:
        The path written.
    """
    config = result.config
    top_n = int(config.output.get("top_n_gpx", 3))

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="slo-terrain-matcher"'
        ' xmlns="http://www.topografix.com/GPX/1/1"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:schemaLocation="http://www.topografix.com/GPX/1/1'
        ' http://www.topografix.com/GPX/11.xsd">',
        "  <metadata>",
        f"    <name>{xml_escape(config.area.get('name', 'terrain matches'))}"
        " terrain matches</name>",
        f"    <time>{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}</time>",
        "  </metadata>",
    ]

    count = 0
    for target in config.targets:
        for rank, match in enumerate(result.matches.get(target.key, [])[:top_n], 1):
            name = (
                f"{target.key} #{rank}: {match.name} "
                f"({match.length_mi:.2f}mi {match.mean_grade_pct:+.1f}%)"
            )
            description = (
                f"{target.label} — {target.spec}. "
                f"Mean {match.mean_grade_pct:+.2f}%, "
                f"max {match.steepest_grade_pct:+.1f}% over "
                f"{float(config.analysis.get('min_window_m', 100)):.0f} m, "
                f"gain {match.gain_m * FEET_PER_METRE:.0f} ft, "
                f"surface {match.surface}, score {match.score:.1f}."
            )
            if match.unreliable_reasons:
                description += " Caveats: " + "; ".join(match.unreliable_reasons) + "."

            parts.append("  <trk>")
            parts.append(f"    <name>{xml_escape(name)}</name>")
            parts.append(f"    <desc>{xml_escape(description)}</desc>")
            parts.append(f"    <type>{xml_escape(target.key)}</type>")
            parts.append("    <trkseg>")

            profile = _profile_for(result, match.corridor_id)
            for index in range(match.start_index, match.end_index + 1):
                lon, lat = match.coords[index - match.start_index]
                if profile is not None:
                    elevation = float(profile.elev_m[index])
                    parts.append(
                        f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}">'
                        f"<ele>{elevation:.1f}</ele></trkpt>"
                    )
                else:  # pragma: no cover - profile is always present
                    parts.append(
                        f'      <trkpt lat="{lat:.7f}" lon="{lon:.7f}"/>'
                    )
            parts.append("    </trkseg>")
            parts.append("  </trk>")
            count += 1

    parts.append("</gpx>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    LOG.info("wrote %s (%d tracks)", path.name, count)
    return path


def _profile_for(result: AnalysisResult, corridor_id: int):
    """Find the profile belonging to a corridor id, or ``None``."""
    for profile in result.profiles:
        if profile.corridor.corridor_id == corridor_id:
            return profile
    return None


def write_map(result: AnalysisResult, path: Path) -> Path:
    """Write a Folium map of the top matches, coloured by feature.

    Args:
        result: The analysis result.
        path: Destination file.

    Returns:
        The path written.
    """
    config = result.config
    top_n = int(config.output.get("top_n_map", 10))
    ref_lat, ref_lon = config.reference_point

    fmap = folium.Map(
        location=[ref_lat, ref_lon], zoom_start=12, tiles="OpenStreetMap"
    )
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap (CC-BY-SA)",
        name="Topographic",
        overlay=False,
    ).add_to(fmap)

    folium.Marker(
        [ref_lat, ref_lon],
        tooltip=f"Reference point: {config.area.get('name', '')}",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(fmap)

    drawn = 0
    for target in config.targets:
        matches = result.matches.get(target.key, [])[:top_n]
        if not matches:
            continue
        group = folium.FeatureGroup(
            name=f"{target.label} ({len(matches)})", show=True
        )
        for rank, match in enumerate(matches, 1):
            popup_html = _popup_html(match, target, rank)
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in match.coords],
                color=target.color,
                weight=6 if rank <= 3 else 4,
                opacity=0.85 if rank <= 3 else 0.6,
                dash_array="8,6" if match.unreliable else None,
                tooltip=f"{target.label} #{rank}: {match.name}",
                popup=folium.Popup(popup_html, max_width=360),
            ).add_to(group)
            folium.CircleMarker(
                location=(match.start_lat, match.start_lon),
                radius=5,
                color=target.color,
                fill=True,
                fill_opacity=1.0,
                tooltip=f"Start of {target.label} #{rank}",
            ).add_to(group)
            drawn += 1
        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    _add_legend(fmap, config)

    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(path))
    LOG.info("wrote %s (%d segments)", path.name, drawn)
    return path


def _popup_html(match: Match, target: Target, rank: int) -> str:
    """Build the popup body for one segment."""
    loop = _loop_text(match, target)
    flags = "; ".join(match.unreliable_reasons)
    rows = [
        ("Length", f"{match.length_mi:.2f} mi"),
        ("Mean grade", f"{match.mean_grade_pct:+.2f}%"),
        ("Max grade", f"{match.steepest_grade_pct:+.1f}% over 100 m"),
        ("Gain", f"{match.gain_m * FEET_PER_METRE:.0f} ft"),
        ("Surface", match.surface),
        ("Loop", loop),
        ("From reference", f"{match.distance_from_reference_mi:.1f} mi"),
        ("Score", f"{match.score:.1f}"),
    ]
    body = "".join(
        f"<tr><td style='padding-right:8px;color:#555'>{html.escape(label)}</td>"
        f"<td><b>{html.escape(value)}</b></td></tr>"
        for label, value in rows
    )
    link = geo.google_maps_link(match.start_lat, match.start_lon)
    warning = (
        f"<p style='margin:6px 0 0;color:#b35c00'><b>Caveat:</b> "
        f"{html.escape(flags)}</p>"
        if flags
        else ""
    )
    return (
        f"<div style='font:13px/1.4 system-ui,sans-serif'>"
        f"<h4 style='margin:0 0 4px'>{html.escape(target.label)} #{rank}</h4>"
        f"<p style='margin:0 0 6px'>{html.escape(match.name)}</p>"
        f"<table style='border-collapse:collapse'>{body}</table>"
        f"{warning}"
        f"<p style='margin:6px 0 0'>"
        f"<a href='{link}' target='_blank' rel='noopener'>Open start in Google Maps</a>"
        f"</p></div>"
    )


def _add_legend(fmap, config: Config) -> None:
    """Attach a fixed-position colour legend to the map."""
    items = "".join(
        f"<div style='margin:2px 0'>"
        f"<span style='display:inline-block;width:14px;height:4px;"
        f"background:{target.color};vertical-align:middle;margin-right:6px'></span>"
        f"{html.escape(target.label)}</div>"
        for target in config.targets
    )
    legend = (
        "<div style=\"position:fixed;bottom:24px;left:12px;z-index:9999;"
        "background:rgba(255,255,255,.94);padding:10px 12px;border-radius:6px;"
        "box-shadow:0 1px 6px rgba(0,0,0,.3);font:12px/1.4 system-ui,sans-serif\">"
        "<b>Course features</b>"
        f"{items}"
        "<div style='margin-top:6px;color:#666'>Dashed = DEM unreliable "
        "(bridge, tunnel or creek)</div></div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend))


def write_all(result: AnalysisResult) -> ReportPaths:
    """Generate every output file.

    Args:
        result: The analysis result.

    Returns:
        The paths of the four generated files.
    """
    out = result.config.output_dir
    return ReportPaths(
        csv=write_csv(result, out / "matches.csv"),
        markdown=write_markdown(result, out / "report.md"),
        gpx=write_gpx(result, out / "routes.gpx"),
        map=write_map(result, out / "map.html"),
    )
