"""Tests for resolving a search area and for runtime config overrides.

Geocoding hits the network, so it is not tested here; everything that turns a
user's input into a bounding box is.
"""

from __future__ import annotations

import math

import pytest

from area import (
    MAX_AREA_SQ_DEG,
    Area,
    bbox_from_center,
    looks_like_united_states,
    resolve_area,
)
from config import load_config


# ---------------------------------------------------------------------------
# Building a box around a point
# ---------------------------------------------------------------------------


def test_bbox_from_center_is_centred():
    west, south, east, north = bbox_from_center(35.28, -120.66, 8.0)
    assert (west + east) / 2 == pytest.approx(-120.66)
    assert (south + north) / 2 == pytest.approx(35.28)


def test_bbox_from_center_has_the_requested_radius():
    """The box should be about 16 miles across for an 8 mile radius."""
    area = Area(
        bbox=bbox_from_center(35.28, -120.66, 8.0),
        center=(35.28, -120.66),
        label="test",
    )
    assert area.height_mi == pytest.approx(16.0, abs=0.2)
    assert area.width_mi == pytest.approx(16.0, abs=0.3)


def test_bbox_longitude_widens_with_latitude():
    """A degree of longitude shrinks toward the poles, so the box widens."""
    equator = bbox_from_center(0.0, 0.0, 10.0)
    north = bbox_from_center(60.0, 0.0, 10.0)
    equator_width = equator[2] - equator[0]
    north_width = north[2] - north[0]
    assert north_width > equator_width
    assert north_width == pytest.approx(equator_width / math.cos(math.radians(60)), rel=0.01)


def test_bbox_rejects_a_bad_radius():
    with pytest.raises(ValueError):
        bbox_from_center(35.0, -120.0, 0.0)
    with pytest.raises(ValueError):
        bbox_from_center(35.0, -120.0, -5.0)


def test_bbox_clamps_latitude_at_the_pole():
    _, south, _, north = bbox_from_center(89.9, 0.0, 50.0)
    assert north <= 90.0 and south >= -90.0


# ---------------------------------------------------------------------------
# resolve_area
# ---------------------------------------------------------------------------


def test_resolve_explicit_bbox_is_used_as_given():
    area = resolve_area(bbox="-120.75,35.15,-120.55,35.35")
    assert area.bbox == (-120.75, 35.15, -120.55, 35.35)
    assert area.center == pytest.approx((35.25, -120.65))


def test_resolve_center_builds_a_box():
    area = resolve_area(center="35.28,-120.66", radius_mi=5.0)
    assert area.radius_mi == 5.0
    assert area.height_mi == pytest.approx(10.0, abs=0.2)


def test_resolve_requires_one_of_the_three():
    with pytest.raises(ValueError, match="give one of"):
        resolve_area()


def test_resolve_rejects_an_inverted_bbox():
    with pytest.raises(ValueError, match="inverted"):
        resolve_area(bbox="-120.55,35.35,-120.75,35.15")


def test_resolve_rejects_a_malformed_bbox():
    with pytest.raises(ValueError, match="four comma-separated"):
        resolve_area(bbox="-120.75,35.15,-120.55")
    with pytest.raises(ValueError, match="non-numeric"):
        resolve_area(bbox="-120.75,35.15,-120.55,north")


def test_resolve_rejects_a_malformed_center():
    with pytest.raises(ValueError, match="lat,lon"):
        resolve_area(center="35.28")
    with pytest.raises(ValueError, match="non-numeric"):
        resolve_area(center="here,there")


def test_resolve_rejects_an_out_of_range_center():
    with pytest.raises(ValueError, match="out of range"):
        resolve_area(center="935.28,-120.66")


def test_resolve_rejects_an_implausibly_large_area():
    """A radius that would cover a whole state is refused, not attempted.

    An Overpass query over that area would hang rather than fail cleanly.
    """
    with pytest.raises(ValueError, match="too large"):
        resolve_area(center="35.28,-120.66", radius_mi=200.0)


def test_area_size_limit_is_consistent_with_the_constant():
    area = resolve_area(center="35.28,-120.66", radius_mi=8.0)
    assert area.area_sq_deg < MAX_AREA_SQ_DEG


def test_area_describe_mentions_the_label_and_size():
    area = resolve_area(center="35.28,-120.66", radius_mi=4.0)
    text = area.describe()
    assert "35.2800" in text
    assert "mi" in text


# ---------------------------------------------------------------------------
# 3DEP coverage check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lat,lon,inside",
    [
        (35.28, -120.66, True),  # San Luis Obispo
        (40.01, -105.27, True),  # Boulder
        (42.36, -71.06, True),  # Boston
        (61.22, -149.90, True),  # Anchorage
        (21.31, -157.86, True),  # Honolulu
        (51.51, -0.13, False),  # London
        (-33.87, 151.21, False),  # Sydney
        (48.86, 2.35, False),  # Paris
        (19.43, -99.13, False),  # Mexico City
    ],
)
def test_us_coverage_check(lat, lon, inside):
    """Elevation is US-only, so a course abroad must be caught early."""
    area = Area(bbox=bbox_from_center(lat, lon, 5.0), center=(lat, lon), label="x")
    assert looks_like_united_states(area) is inside


# ---------------------------------------------------------------------------
# Runtime config overrides
# ---------------------------------------------------------------------------


def test_with_area_retargets_without_mutating_the_original():
    config = load_config()
    original_bbox = config.bbox
    moved = config.with_area(
        (-105.4, 39.9, -105.1, 40.1), center=(40.0, -105.25), name="Boulder, CO"
    )
    assert moved.bbox == (-105.4, 39.9, -105.1, 40.1)
    assert moved.reference_point == (40.0, -105.25)
    assert moved.area["name"] == "Boulder, CO"
    assert config.bbox == original_bbox, "the original config must be untouched"


def test_with_area_keeps_targets_and_analysis_settings():
    config = load_config()
    moved = config.with_area((-105.4, 39.9, -105.1, 40.1))
    assert [t.key for t in moved.targets] == [t.key for t in config.targets]
    assert moved.analysis == config.analysis


def test_with_course_overrides_only_the_course():
    config = load_config()
    swapped = config.with_course("other.gpx")
    assert swapped.course_gpx == "other.gpx"
    assert swapped.bbox == config.bbox
    assert config.course_gpx != "other.gpx"


def test_with_routes_replaces_the_route_specs():
    config = load_config()
    swapped = config.with_routes([{"key": "only", "length_mi": 5.0}])
    assert [r["key"] for r in swapped.routes] == ["only"]
    assert len(config.routes) != 1 or config.routes[0]["key"] != "only"


def test_cache_key_differs_between_areas():
    """Two areas must not share a cache entry.

    They are keyed by bounding box; if the key collided, a run for one town
    would silently analyse another town's roads.
    """
    from fetch_data import _bbox_key

    slo = _bbox_key((-120.75, 35.15, -120.55, 35.35))
    boulder = _bbox_key((-105.4, 39.9, -105.1, 40.1))
    assert slo != boulder
    assert _bbox_key((-120.75, 35.15, -120.55, 35.35)) == slo
