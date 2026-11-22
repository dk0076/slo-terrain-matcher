"""Chart a replica's profile against the course it was built from.

The summary line reports a single correlation number. That tells you *how
well* the replica matches, but not *where* it stops matching, which is the
thing you can actually act on: a replica that nails the first twenty miles and
falls apart in the finish is a different problem from one that is uniformly
half a percent off.

Both series are plotted against **fraction of the way through** rather than
absolute miles, because that is what the route search matches on. For a
shorter replica the course's elevation is scaled by the length ratio, which is
the same thing as compressing the course while keeping its grades — so the two
curves stay directly comparable and the gap between them is genuinely extra
(or missing) climbing rather than an artefact of the distance.

Output is a self-contained HTML file with inline SVG. No plotting library and
no network fetch, so it opens anywhere.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOG = logging.getLogger("chart")

#: Metres to feet.
FEET_PER_METRE = 3.280839895

#: Plot geometry.
WIDTH = 900
PANEL_HEIGHT = 230
LEFT = 70
RIGHT = 30
TOP = 34
GAP = 64

#: Grade difference, in percentage points, above which a stretch is called out
#: as diverging. Roughly the point at which a runner notices.
DIVERGENCE_PCT = 1.5

COURSE_COLOR = "#8a8880"
REPLICA_COLOR = "#2a78d6"
DIVERGE_COLOR = "#eb6834"


def _scale(value, lo, hi, out_lo, out_hi):
    """Map a value from one range onto another, guarding a zero-width input."""
    if hi - lo < 1e-9:
        return (out_lo + out_hi) / 2.0
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def _polyline(xs, ys, color, width=2.0, opacity=1.0, dash=None):
    """One SVG polyline."""
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round" '
        f'opacity="{opacity}"{dash_attr} points="{points}"/>'
    )


def _panel(
    title: str,
    y_top: float,
    fractions: np.ndarray,
    course: np.ndarray,
    replica: np.ndarray,
    y_label: str,
    diverge_mask: np.ndarray | None = None,
) -> str:
    """Render one comparison panel as SVG."""
    plot_left = LEFT
    plot_right = WIDTH - RIGHT
    plot_top = y_top
    plot_bottom = y_top + PANEL_HEIGHT

    combined = np.concatenate([course, replica])
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))
    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    parts: list[str] = []
    parts.append(
        f'<text x="{plot_left}" y="{plot_top - 12}" class="panel-title">'
        f"{html.escape(title)}</text>"
    )

    # Horizontal gridlines with labels.
    ticks = np.linspace(lo, hi, 5)
    for tick in ticks:
        y = _scale(tick, lo, hi, plot_bottom, plot_top)
        parts.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{plot_left - 10}" y="{y + 4:.1f}" class="tick" '
            f'text-anchor="end">{tick:,.0f}</text>'
        )

    # Zero line, when the range crosses it.
    if lo < 0 < hi:
        zero_y = _scale(0.0, lo, hi, plot_bottom, plot_top)
        parts.append(
            f'<line x1="{plot_left}" y1="{zero_y:.1f}" x2="{plot_right}" '
            f'y2="{zero_y:.1f}" class="zero"/>'
        )

    # Vertical gridlines every 10%.
    for pct in range(0, 101, 10):
        x = _scale(pct / 100.0, 0.0, 1.0, plot_left, plot_right)
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" '
            f'y2="{plot_bottom}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 18:.1f}" class="tick" '
            f'text-anchor="middle">{pct}%</text>'
        )

    xs = [_scale(f, 0.0, 1.0, plot_left, plot_right) for f in fractions]

    # Shade the stretches that diverge.
    if diverge_mask is not None and diverge_mask.any():
        start = None
        for index, flag in enumerate(diverge_mask):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                parts.append(
                    f'<rect x="{xs[start]:.1f}" y="{plot_top}" '
                    f'width="{max(1.0, xs[index - 1] - xs[start]):.1f}" '
                    f'height="{PANEL_HEIGHT}" class="diverge"/>'
                )
                start = None
        if start is not None:
            parts.append(
                f'<rect x="{xs[start]:.1f}" y="{plot_top}" '
                f'width="{max(1.0, xs[-1] - xs[start]):.1f}" '
                f'height="{PANEL_HEIGHT}" class="diverge"/>'
            )

    course_ys = [_scale(v, lo, hi, plot_bottom, plot_top) for v in course]
    replica_ys = [_scale(v, lo, hi, plot_bottom, plot_top) for v in replica]
    parts.append(_polyline(xs, course_ys, COURSE_COLOR, 2.0, 0.95))
    parts.append(_polyline(xs, replica_ys, REPLICA_COLOR, 2.0, 0.95))

    # Axis frame.
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
        f'y2="{plot_bottom}" class="axis"/>'
    )
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
        f'y2="{plot_bottom}" class="axis"/>'
    )
    parts.append(
        f'<text x="18" y="{(plot_top + plot_bottom) / 2:.1f}" class="axis-label" '
        f'transform="rotate(-90 18 {(plot_top + plot_bottom) / 2:.1f})" '
        f'text-anchor="middle">{html.escape(y_label)}</text>'
    )
    return "\n".join(parts)


def write_profile_comparison(
    route, course_profile, path: Path, course_name: str = "course"
) -> Path:
    """Write an HTML chart comparing a replica to its course.

    Args:
        route: A built :class:`route.Route`.
        course_profile: The :class:`course.CourseProfile` it was built from.
        path: Destination ``.html`` file.
        course_name: Label for the course series.

    Returns:
        The path written.
    """
    from route import _route_grades

    target = route.target
    total_m = float(route.dist_m[-1])

    # Grades, measured over the course's window scaled to this route's length
    # so the two series are compared at the same physical scale.
    window = max(100.0, target.window_m * (total_m / target.total_m))
    fractions, replica_grades = _route_grades(route.dist_m, route.elevations, window)
    if fractions.size == 0:
        raise ValueError("route is too short to chart")
    course_grades = target.grade_at(fractions)
    difference = replica_grades - course_grades
    diverge = np.abs(difference) > DIVERGENCE_PCT

    # Elevation, each normalised to its own start. The course is scaled by the
    # length ratio, which keeps its grades and makes the curves comparable.
    scale = total_m / target.total_m
    replica_elev_ft = (route.elevations - route.elevations[0]) * FEET_PER_METRE
    replica_fracs = route.dist_m / total_m

    course_fracs = course_profile.dist_m / float(course_profile.dist_m[-1])
    course_elev_ft = (
        (course_profile.elev_m - course_profile.elev_m[0]) * scale * FEET_PER_METRE
    )

    # Resample both onto a shared grid so they can be drawn point for point.
    grid = np.linspace(0.0, 1.0, 600)
    replica_on_grid = np.interp(grid, replica_fracs, replica_elev_ft)
    course_on_grid = np.interp(grid, course_fracs, course_elev_ft)
    grade_diverge_on_grid = np.interp(grid, fractions, diverge.astype(float)) > 0.5

    height = TOP + PANEL_HEIGHT + GAP + PANEL_HEIGHT + 70
    panels = [
        _panel(
            "Grade through the route",
            TOP,
            fractions,
            course_grades,
            replica_grades,
            "grade (%)",
            diverge,
        ),
        _panel(
            "Cumulative elevation change",
            TOP + PANEL_HEIGHT + GAP,
            grid,
            course_on_grid,
            replica_on_grid,
            "feet",
            grade_diverge_on_grid,
        ),
    ]

    fit = route.fit
    diverging_fraction = float(diverge.mean()) * 100.0
    worst_index = int(np.argmax(np.abs(difference)))
    worst_at = float(fractions[worst_index]) * 100.0
    worst_by = float(difference[worst_index])

    svg = (
        f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Grade and elevation of the replica compared with the course">'
        + "\n".join(panels)
        + f'<text x="{WIDTH / 2}" y="{height - 26}" class="axis-label" '
        f'text-anchor="middle">fraction of the way through</text>'
        + "</svg>"
    )

    stats = [
        ("Replica length", f"{route.length_mi:.2f} mi"),
        ("Course length", f"{target.total_m / 1609.344:.2f} mi"),
        ("Replica climbing", f"{route.gain_m * FEET_PER_METRE:,.0f} ft"),
        (
            "Course over the same distance",
            f"{fit.get('course_equivalent_gain_ft', 0):,.0f} ft",
        ),
        ("Grade error", f"{fit.get('grade_mae_pct', 0):.2f} pts"),
        ("Profile match", f"{fit.get('grade_correlation', float('nan')):.2f}"),
        ("Diverging stretches", f"{diverging_fraction:.0f}% of the route"),
        (
            "Worst divergence",
            f"{worst_by:+.1f} pts at {worst_at:.0f}% through",
        ),
    ]
    stat_html = "".join(
        f'<div class="stat"><span class="k">{html.escape(k)}</span>'
        f'<span class="v">{html.escape(v)}</span></div>'
        for k, v in stats
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(route.name)} vs {html.escape(course_name)}</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #1a1a19; --muted: #6b6a65;
    --grid: #e8e7e1; --axis: #b4b2a9; --card: #faf9f6; --border: #e2e0d8;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #141413; --fg: #f0efec; --muted: #9c9a92;
      --grid: #2a2a28; --axis: #44443f; --card: #1d1d1b; --border: #2e2e2b;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 28px 20px 48px; background: var(--bg); color: var(--fg);
    font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 21px; font-weight: 500; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); font-size: 14px; margin: 0 0 20px; }}
  .legend {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 0 0 8px; font-size: 13px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
  .swatch {{ width: 16px; height: 3px; border-radius: 2px; display: inline-block; }}
  .swatch.box {{ height: 12px; width: 12px; border-radius: 2px; opacity: .3; }}
  svg {{ display: block; margin: 0 0 20px; }}
  .panel-title {{ font: 500 14px system-ui, sans-serif; fill: var(--fg); }}
  .tick {{ font: 11px system-ui, sans-serif; fill: var(--muted); }}
  .axis-label {{ font: 12px system-ui, sans-serif; fill: var(--muted); }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .zero {{ stroke: var(--axis); stroke-width: 1; stroke-dasharray: 4 4; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .diverge {{ fill: {DIVERGE_COLOR}; opacity: .10; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 10px; margin: 8px 0 0;
  }}
  .stat {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 13px; display: flex; flex-direction: column; gap: 2px;
  }}
  .stat .k {{ font-size: 12px; color: var(--muted); }}
  .stat .v {{ font-size: 17px; font-weight: 500; }}
  .note {{ color: var(--muted); font-size: 13px; margin: 20px 0 0; max-width: 70ch; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(route.name)}</h1>
  <p class="sub">compared with {html.escape(course_name)} &middot; generated {generated}</p>
  <div class="legend">
    <span><i class="swatch" style="background:{COURSE_COLOR}"></i>the course</span>
    <span><i class="swatch" style="background:{REPLICA_COLOR}"></i>this replica</span>
    <span><i class="swatch box" style="background:{DIVERGE_COLOR}"></i>
      diverging by more than {DIVERGENCE_PCT:g} points</span>
  </div>
  {svg}
  <div class="stats">{stat_html}</div>
  <p class="note">
    Both series are plotted against fraction of the way through, which is what
    the route search matches on. The course's elevation is scaled by the
    length ratio &mdash; the same as compressing the course while keeping its
    grades &mdash; so the gap between the two elevation curves is real extra
    or missing climbing, not an artefact of the two distances differing.
  </p>
</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    LOG.info("wrote %s", path.name)
    return path
