"""A small web front end: upload a course GPX, pick an area, get a route.

Run it with::

    python webapp.py

then open http://127.0.0.1:5000.

Building a route takes anywhere from fifteen seconds to a couple of minutes,
mostly waiting on Overpass, so requests do not block. A submission creates a
job, the browser polls it, and the work happens on a small pool of background
threads.

Two things are deliberately conservative, because this is the kind of app that
is easy to point at the public internet without thinking:

* **Uploads are parsed defensively.** A GPX is XML from a stranger, so the
  file is size-capped and parsed with ``defusedxml`` to shut off entity
  expansion and external entity resolution. Without that, a few kilobytes of
  nested entities can exhaust the server's memory.
* **Work is rate limited and queued.** Every new area costs an Overpass query
  and a DEM read. Overpass is a volunteer service, and hammering it from a
  public deployment is both antisocial and a fast route to being blocked, so
  the queue is deliberately narrow and repeat areas are served from cache.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
)

from area import looks_like_united_states, resolve_area
from config import configure_logging, load_config

LOG = logging.getLogger("webapp")

#: Largest GPX accepted. A marathon course with a point every few metres is
#: well under a megabyte; anything far larger is a mistake or an attack.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

#: How many routes may be built at once. Each one holds a DEM in memory and
#: may hit Overpass, so this stays small on purpose.
MAX_WORKERS = 2

#: Jobs kept before the oldest are discarded.
MAX_JOBS = 200

#: Per-client limits.
RATE_LIMIT_REQUESTS = 5
RATE_LIMIT_WINDOW_S = 300

#: Bounds on what a caller may ask for.
MAX_SCENIC = 3.0
MIN_MILES, MAX_MILES = 1.0, 50.0
MIN_RADIUS_MI, MAX_RADIUS_MI = 1.0, 15.0

#: Where uploads and results live.
WORK_DIR = Path(os.environ.get("TERRAIN_WORK_DIR", "web-work"))


@dataclass
class Job:
    """One route build."""

    id: str
    state: str = "queued"  # queued | running | done | failed
    stage: str = "waiting to start"
    progress: int = 0
    error: str | None = None
    result: dict[str, Any] | None = None
    created: float = field(default_factory=time.time)
    gpx_path: Path | None = None
    chart_path: Path | None = None
    course_path: Path | None = None
    course_name: str = "course"

    def public(self) -> dict[str, Any]:
        """The view of this job handed to the browser."""
        return {
            "id": self.id,
            "state": self.state,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "result": self.result,
        }


_jobs: dict[str, Job] = {}
_job_order: deque[str] = deque()
_jobs_lock = threading.Lock()
_hits: dict[str, deque[float]] = {}
_hits_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="build")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.secret_key = os.environ.get("TERRAIN_SECRET_KEY") or secrets.token_hex(16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_key() -> str:
    """Identify the caller for rate limiting.

    Honours ``X-Forwarded-For`` only when a proxy is declared, so a client
    cannot simply claim a different address to reset its own limit.
    """
    if os.environ.get("TERRAIN_BEHIND_PROXY") == "1":
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _over_limit() -> bool:
    """Whether this caller has used up its allowance.

    Checking and consuming are separate on purpose. A rejected request costs
    the server nothing, so charging it against the allowance would let a
    mistyped distance or a wrong file lock someone out for minutes without
    any work ever having been done.
    """
    now = time.time()
    with _hits_lock:
        bucket = _hits.setdefault(_client_key(), deque())
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_S:
            bucket.popleft()
        return len(bucket) >= RATE_LIMIT_REQUESTS


def _record_hit() -> None:
    """Charge one accepted build against this caller's allowance."""
    with _hits_lock:
        _hits.setdefault(_client_key(), deque()).append(time.time())


def _remember(job: Job) -> None:
    """Store a job, discarding the oldest once the cap is reached."""
    with _jobs_lock:
        _jobs[job.id] = job
        _job_order.append(job.id)
        while len(_job_order) > MAX_JOBS:
            stale = _job_order.popleft()
            old = _jobs.pop(stale, None)
            if old is not None:
                import shutil

                try:
                    shutil.rmtree(WORK_DIR / old.id, ignore_errors=True)
                except OSError:  # pragma: no cover - best effort
                    pass


def _validate_gpx(path: Path) -> str:
    """Check an uploaded file really is a GPX track, safely.

    Args:
        path: The saved upload.

    Returns:
        The track name, or an empty string.

    Raises:
        ValueError: If the file is not parseable GPX or has no track points.
    """
    from defusedxml.ElementTree import parse as safe_parse

    try:
        tree = safe_parse(str(path))
    except Exception as error:  # noqa: BLE001 - any parse failure is a reject
        raise ValueError(
            "that file could not be read as GPX. Export the course as a GPX "
            "track and try again."
        ) from error

    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag[: root.tag.index("}") + 1]
    if not root.tag.endswith("gpx"):
        raise ValueError("that file is XML, but not a GPX file.")

    points = root.findall(f".//{namespace}trkpt")
    if len(points) < 10:
        raise ValueError(
            f"that GPX has {len(points)} track points. A course needs a "
            "recorded or drawn track, not just waypoints."
        )
    with_elevation = sum(
        1 for point in points if point.find(f"{namespace}ele") is not None
    )
    if with_elevation < len(points) * 0.5:
        raise ValueError(
            "that GPX has no elevation data, so there is no profile to "
            "reproduce. Export it with elevation included."
        )

    name_node = root.find(f".//{namespace}trk/{namespace}name")
    return (name_node.text or "").strip() if name_node is not None else ""


def _safe_stem(text: str) -> str:
    """A filesystem-safe stem from arbitrary text.

    The result becomes a path component and a ``Content-Disposition``
    filename, so it must not be able to carry path traversal or resolve to a
    directory reference. Everything outside a conservative alphabet is
    collapsed to a hyphen, and runs of dots are removed rather than kept, so
    ``..`` and ``...`` cannot survive in any position.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    cleaned = re.sub(r"\.{2,}", "-", cleaned)  # no .. anywhere
    cleaned = cleaned.strip("-._")
    return (cleaned or "course")[:48].strip("-._") or "course"


# ---------------------------------------------------------------------------
# The build job
# ---------------------------------------------------------------------------


def _run_job(job: Job, params: dict[str, Any]) -> None:
    """Build one route, updating the job as it goes."""
    from replicate import replicate

    try:
        job.state = "running"
        job.stage = "working out where to search"
        job.progress = 5

        area = resolve_area(
            center=f"{params['lat']},{params['lon']}",
            radius_mi=params["radius"],
        )
        if not looks_like_united_states(area):
            raise ValueError(
                "elevation data covers the United States only, so a route "
                "cannot be built there. The course itself can be from "
                "anywhere — only the place you want to run has to be in "
                "the US."
            )

        config = load_config().with_area(
            area.bbox, center=area.center, name=area.label
        )
        def on_progress(percent: int, stage: str) -> None:
            job.progress = max(job.progress, percent)
            job.stage = stage

        miles_label = (
            f"{params['miles']:g} mi" if params["miles"] else "full distance"
        )
        out_path = job.course_path.parent / f"{job.course_name}-replica.gpx"
        built, chart_path = replicate(
            config,
            area,
            job.course_path,
            params["miles"],
            out_path,
            start=(params["lat"], params["lon"]),
            scenic_weight=params.get("scenic", 0.0),
            on_progress=on_progress,
            route_name=f"{job.course_name} replica — {miles_label}",
        )
        if built is None:
            raise ValueError(
                "no route of that length could be built from there. The roads "
                "around that point may be too sparse to close a loop — "
                "try a larger search radius, a different spot, or a shorter "
                "distance."
            )

        job.gpx_path = out_path
        job.chart_path = chart_path
        fit = built.fit
        job.result = {
            "name": built.name,
            "length_mi": round(built.length_mi, 2),
            "gain_ft": round(built.gain_m * 3.280839895),
            "course_gain_ft": round(fit.get("course_equivalent_gain_ft", 0.0)),
            "steepest_climb": round(fit.get("steepest_climb_pct", 0.0), 1),
            "steepest_descent": round(fit.get("steepest_descent_pct", 0.0), 1),
            "course_steepest_climb": round(
                fit.get("course_steepest_climb_pct", 0.0), 1
            ),
            "course_steepest_descent": round(
                fit.get("course_steepest_descent_pct", 0.0), 1
            ),
            "grade_error": round(fit.get("grade_mae_pct", 0.0), 2),
            "correlation": round(fit.get("grade_correlation", 0.0), 2),
            "mean_drift_ft": round(fit.get("mean_drift_ft", 0.0)),
            "max_drift_ft": round(fit.get("max_drift_ft", 0.0)),
            "repeat_pct": round(built.repeat_fraction * 100),
            "pleasantness": (
                round(nice, 2)
                if (nice := built.mean_pleasantness(
                    getattr(built, "pleasantness_scores", None)
                )) is not None
                else None
            ),
            "closed_loop": built.closed_loop,
            "paved_pct": round(built.surface_fractions.get("paved", 0.0) * 100),
            "streets": built.names[:14],
            "area": area.label,
            "has_chart": chart_path is not None,
            "coords": [
                [round(float(lat), 6), round(float(lon), 6)]
                for lon, lat in built.coords[::3]
            ],
        }
        job.stage = "done"
        job.progress = 100
        job.state = "done"
        LOG.info("job %s finished: %.2f mi", job.id, built.length_mi)

    except ValueError as error:
        job.state = "failed"
        job.error = str(error)
        LOG.warning("job %s failed: %s", job.id, error)
    except Exception as error:  # noqa: BLE001 - surface anything else safely
        job.state = "failed"
        job.error = (
            "something went wrong building that route. If it keeps happening, "
            "try a different area or distance."
        )
        LOG.exception("job %s crashed: %s", job.id, error)
    finally:
        try:
            if job.course_path and job.course_path.exists():
                job.course_path.unlink()
                job.course_path = None
        except OSError:  # pragma: no cover - best effort
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    """The single page."""
    return render_template("index.html")


@app.post("/api/build")
def build():
    """Accept a course and an area, and start a job."""
    if _over_limit():
        return (
            jsonify(
                error=(
                    f"that is {RATE_LIMIT_REQUESTS} routes in "
                    f"{RATE_LIMIT_WINDOW_S // 60} minutes. Each one costs a "
                    "fetch from OpenStreetMap's volunteer servers, so please "
                    "give it a few minutes."
                )
            ),
            429,
        )

    upload = request.files.get("course")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a course GPX file first."), 400

    try:
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])
        radius = float(request.form.get("radius", 8))
        miles_raw = request.form.get("miles", "").strip()
        miles = float(miles_raw) if miles_raw else None
        scenic = float(request.form.get("scenic", 0) or 0)
    except (KeyError, ValueError):
        return jsonify(error="Pick a spot on the map and a distance."), 400

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify(error="That map position is not valid."), 400
    if not (MIN_RADIUS_MI <= radius <= MAX_RADIUS_MI):
        return (
            jsonify(
                error=f"Search radius must be between {MIN_RADIUS_MI:g} and "
                f"{MAX_RADIUS_MI:g} miles."
            ),
            400,
        )
    if not (0.0 <= scenic <= MAX_SCENIC):
        return (
            jsonify(error=f"Scenic preference must be between 0 and {MAX_SCENIC:g}."),
            400,
        )
    if miles is not None and not (MIN_MILES <= miles <= MAX_MILES):
        return (
            jsonify(
                error=f"Distance must be between {MIN_MILES:g} and "
                f"{MAX_MILES:g} miles."
            ),
            400,
        )

    job_id = uuid.uuid4().hex[:12]
    # Each job gets its own directory, so the uploaded course keeps its real
    # name. That name ends up in the route title and the download, and a job
    # id baked into it looks like a bug to whoever downloads the file.
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    course_name = _safe_stem(Path(upload.filename).stem)
    course_path = job_dir / f"{course_name}.gpx"
    upload.save(course_path)

    try:
        _validate_gpx(course_path)
    except ValueError as error:
        course_path.unlink(missing_ok=True)
        return jsonify(error=str(error)), 400

    job = Job(id=job_id, course_path=course_path, course_name=course_name)
    _record_hit()
    _remember(job)
    _pool.submit(
        _run_job,
        job,
        {
            "lat": lat,
            "lon": lon,
            "radius": radius,
            "miles": miles,
            "scenic": scenic,
        },
    )
    LOG.info("job %s queued (%.4f, %.4f, %.0f mi radius)", job_id, lat, lon, radius)
    return jsonify(job.public()), 202


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    """Poll a job."""
    job = _jobs.get(job_id)
    if job is None:
        return jsonify(error="That job has expired."), 404
    return jsonify(job.public())


@app.get("/api/job/<job_id>/gpx")
def job_gpx(job_id: str):
    """Download the finished route."""
    job = _jobs.get(job_id)
    if job is None or not job.gpx_path or not job.gpx_path.exists():
        return jsonify(error="That route is no longer available."), 404
    return send_file(
        job.gpx_path,
        mimetype="application/gpx+xml",
        as_attachment=True,
        download_name=f"{job.course_name}-replica.gpx",
    )


@app.get("/api/job/<job_id>/chart")
def job_chart(job_id: str):
    """The comparison chart for a finished route."""
    job = _jobs.get(job_id)
    if job is None or not job.chart_path or not job.chart_path.exists():
        return jsonify(error="No chart for that route."), 404
    return send_file(job.chart_path, mimetype="text/html")


@app.errorhandler(413)
def too_large(_error):
    """A friendlier message than Flask's default for an oversized upload."""
    return (
        jsonify(
            error=f"That file is larger than "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. A course GPX should be "
            "well under a megabyte."
        ),
        413,
    )


def main() -> int:
    """Run the development server."""
    configure_logging()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("TERRAIN_HOST", "127.0.0.1")
    port = int(os.environ.get("TERRAIN_PORT", "5000"))
    LOG.info("open http://%s:%d", host, port)
    app.run(host=host, port=port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
