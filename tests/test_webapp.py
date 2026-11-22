"""Tests for the web front end's validation, limits and job handling.

These cover everything up to the point where a route would actually be built.
The build itself needs the network and several minutes, so it is exercised by
running the app, not here.
"""

from __future__ import annotations

import io

import pytest

import webapp


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A test client with isolated state and the rate limit cleared."""
    monkeypatch.setattr(webapp, "WORK_DIR", tmp_path / "work")
    webapp._hits.clear()
    webapp._jobs.clear()
    webapp._job_order.clear()
    webapp.app.config.update(TESTING=True)
    with webapp.app.test_client() as test_client:
        yield test_client


def _gpx(points: int = 50, elevation: bool = True) -> bytes:
    """A minimal valid GPX track."""
    body = []
    for index in range(points):
        ele = f"<ele>{10 + index * 0.5:.1f}</ele>" if elevation else ""
        body.append(
            f'<trkpt lat="{35.0 + index * 0.001:.6f}" '
            f'lon="{-120.0 + index * 0.001:.6f}">{ele}</trkpt>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        f'<trk><name>Test</name><trkseg>{"".join(body)}</trkseg></trk></gpx>'
    ).encode()


def _post(client, data=None, course=_gpx(), filename="course.gpx"):
    """Submit a build request with sensible defaults."""
    payload = {"lat": "35.2828", "lon": "-120.6596", "radius": "8", "miles": "9"}
    payload.update(data or {})
    if course is not None:
        payload["course"] = (io.BytesIO(course), filename)
    return client.post("/api/build", data=payload, content_type="multipart/form-data")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Run the race before the race" in response.data


def test_index_credits_the_data_sources(client):
    """OpenStreetMap's licence requires attribution wherever its data shows."""
    body = client.get("/").data
    assert b"openstreetmap.org/copyright" in body
    assert b"3d-elevation-program" in body


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------


def test_missing_file_is_rejected(client):
    response = _post(client, course=None)
    assert response.status_code == 400
    assert "GPX" in response.get_json()["error"]


def test_non_xml_is_rejected(client):
    response = _post(client, course=b"this is not xml at all")
    assert response.status_code == 400
    assert "could not be read as GPX" in response.get_json()["error"]


def test_xml_that_is_not_gpx_is_rejected(client):
    response = _post(client, course=b'<?xml version="1.0"?><root><a/></root>')
    assert response.status_code == 400
    assert "not a GPX file" in response.get_json()["error"]


def test_gpx_without_track_points_is_rejected(client):
    course = (
        '<?xml version="1.0"?>'
        '<gpx xmlns="http://www.topografix.com/GPX/1/1">'
        '<wpt lat="1" lon="2"/></gpx>'
    ).encode()
    response = _post(client, course=course)
    assert response.status_code == 400
    assert "track points" in response.get_json()["error"]


def test_gpx_without_elevation_is_rejected(client):
    """No elevation means no profile to reproduce, so say so plainly."""
    response = _post(client, course=_gpx(elevation=False))
    assert response.status_code == 400
    assert "elevation" in response.get_json()["error"]


def test_an_xml_bomb_does_not_expand(client):
    """Entity expansion must be off, or a few KB can exhaust the server.

    This is the reason uploads go through defusedxml rather than the standard
    library parser.
    """
    bomb = (
        '<?xml version="1.0"?>'
        "<!DOCTYPE lolz ["
        '<!ENTITY lol "lol">'
        '<!ENTITY lol2 "' + "&lol;" * 10 + '">'
        '<!ENTITY lol3 "' + "&lol2;" * 10 + '">'
        "]>"
        "<gpx><trk>&lol3;</trk></gpx>"
    ).encode()
    response = _post(client, course=bomb)
    assert response.status_code == 400


def test_an_external_entity_is_not_resolved(client):
    """A course file must not be able to read files off the server."""
    xxe = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        "<gpx><trk><name>&xxe;</name></trk></gpx>"
    ).encode()
    response = _post(client, course=xxe)
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("lat", "999", "map position"),
        ("lon", "-500", "map position"),
        ("radius", "99", "Search radius"),
        ("radius", "0.1", "Search radius"),
        ("miles", "500", "Distance"),
        ("miles", "0.1", "Distance"),
        ("lat", "not-a-number", "Pick a spot"),
    ],
)
def test_out_of_range_parameters_are_rejected(client, field, value, fragment):
    response = _post(client, {field: value})
    assert response.status_code == 400
    assert fragment in response.get_json()["error"]


def test_missing_position_is_rejected(client):
    response = client.post(
        "/api/build",
        data={"course": (io.BytesIO(_gpx()), "c.gpx")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Pick a spot" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rejected_requests_do_not_consume_the_allowance(client):
    """A bad file should never lock someone out.

    Rejections cost the server nothing, so charging them against the limit
    would let a few fumbled uploads block a legitimate one.
    """
    for _ in range(webapp.RATE_LIMIT_REQUESTS + 3):
        assert _post(client, course=b"nope").status_code == 400

    # A valid request must still be accepted afterwards.
    assert _post(client).status_code == 202


def test_accepted_requests_do_consume_the_allowance(client):
    for _ in range(webapp.RATE_LIMIT_REQUESTS):
        assert _post(client).status_code == 202
    response = _post(client)
    assert response.status_code == 429
    assert "OpenStreetMap" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def test_a_valid_submission_creates_a_job(client):
    response = _post(client)
    assert response.status_code == 202
    body = response.get_json()
    assert body["state"] in {"queued", "running"}
    assert body["id"]


def test_unknown_job_is_a_clean_404(client):
    response = client.get("/api/job/doesnotexist")
    assert response.status_code == 404
    assert "expired" in response.get_json()["error"]


def test_downloading_an_unbuilt_route_is_a_clean_404(client):
    job_id = _post(client).get_json()["id"]
    response = client.get(f"/api/job/{job_id}/gpx")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("boston marathon", "boston-marathon"),
        ("../../etc/passwd", "etc-passwd"),
        ("course (2)", "course-2"),
        ("", "course"),
        ("...", "course"),
    ],
)
def test_safe_stem_strips_anything_dangerous(raw, expected):
    """Upload names reach the filesystem and the download header."""
    assert webapp._safe_stem(raw) == expected


def test_safe_stem_is_length_capped():
    assert len(webapp._safe_stem("x" * 500)) <= 48
