"""Resolve a search area from a place name, a centre point, or a raw bbox.

The pipeline needs a bounding box. People think in terms of "San Luis Obispo"
or "within 8 miles of my house", so this module turns those into one.

A deliberate choice: when a place name is geocoded, the area is built from the
match's **centre point plus a radius**, not from the geocoder's own bounding
box. Geocoders resolve "San Luis Obispo, CA" to the entire county -- roughly
two degrees of latitude -- which would mean an Overpass query covering
thousands of square miles and a DEM mosaic spanning several tiles. A radius
around the centre is predictable and is what someone picking a place to run
actually means.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import requests

LOG = logging.getLogger("area")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

USER_AGENT = "slo-terrain-matcher/1.0 (local training-route analysis)"

#: Degrees of latitude per statute mile.
DEG_LAT_PER_MILE = 1.0 / 69.047

#: Refuse to build an area larger than this, in square degrees. Overpass
#: queries and DEM mosaics both grow with area, and a box this large already
#: covers far more ground than anyone will run.
MAX_AREA_SQ_DEG = 0.75

#: Warn above this.
WARN_AREA_SQ_DEG = 0.25


@dataclass(frozen=True)
class Area:
    """A resolved search area."""

    bbox: tuple[float, float, float, float]  # west, south, east, north
    center: tuple[float, float]  # lat, lon
    label: str
    radius_mi: float | None = None

    @property
    def width_mi(self) -> float:
        """Approximate east-west extent in miles."""
        west, _, east, _ = self.bbox
        lat = self.center[0]
        return (east - west) * 69.047 * math.cos(math.radians(lat))

    @property
    def height_mi(self) -> float:
        """Approximate north-south extent in miles."""
        _, south, _, north = self.bbox
        return (north - south) * 69.047

    @property
    def area_sq_deg(self) -> float:
        """Bounding box area in square degrees."""
        west, south, east, north = self.bbox
        return (east - west) * (north - south)

    def describe(self) -> str:
        """A one-line human description of the area."""
        return (
            f"{self.label} — {self.width_mi:.1f} x {self.height_mi:.1f} mi "
            f"around {self.center[0]:.4f}, {self.center[1]:.4f}"
        )


def bbox_from_center(
    lat: float, lon: float, radius_mi: float
) -> tuple[float, float, float, float]:
    """Build a bounding box of ``radius_mi`` around a point.

    Args:
        lat: Centre latitude in degrees.
        lon: Centre longitude in degrees.
        radius_mi: Half-width of the box in statute miles.

    Returns:
        ``(west, south, east, north)`` in EPSG:4326.

    Raises:
        ValueError: If ``radius_mi`` is not positive.
    """
    if radius_mi <= 0:
        raise ValueError("radius_mi must be positive")

    d_lat = radius_mi * DEG_LAT_PER_MILE
    # Guard against the cosine collapsing near the poles.
    cos_lat = max(0.05, math.cos(math.radians(lat)))
    d_lon = radius_mi * DEG_LAT_PER_MILE / cos_lat

    return (
        lon - d_lon,
        max(-90.0, lat - d_lat),
        lon + d_lon,
        min(90.0, lat + d_lat),
    )


def geocode(query: str, timeout: int = 60) -> tuple[float, float, str]:
    """Look up a place name and return its centre point.

    Args:
        query: A place name, e.g. ``"San Luis Obispo, CA"``.
        timeout: Request timeout in seconds.

    Returns:
        ``(latitude, longitude, display_name)``.

    Raises:
        LookupError: If the geocoder returns no match.
        requests.HTTPError: If the request fails.
    """
    LOG.info("geocoding %r", query)
    response = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise LookupError(
            f"no place matched {query!r}; try adding a state or country, "
            "or pass --center lat,lon instead"
        )

    top = results[0]
    lat, lon = float(top["lat"]), float(top["lon"])
    name = str(top.get("display_name", query))
    LOG.info("  matched %s (%.4f, %.4f)", name, lat, lon)
    # Nominatim asks for at most one request per second.
    time.sleep(1.0)
    return lat, lon, name


def resolve_area(
    place: str | None = None,
    center: str | None = None,
    bbox: str | None = None,
    radius_mi: float = 8.0,
) -> Area:
    """Resolve an area from whichever option the caller supplied.

    Exactly one of ``place``, ``center`` or ``bbox`` should be given. A raw
    ``bbox`` is used as-is; the other two build a box of ``radius_mi`` around
    a point.

    Args:
        place: A place name to geocode.
        center: ``"lat,lon"``.
        bbox: ``"west,south,east,north"``.
        radius_mi: Half-width of the box for ``place`` and ``center``.

    Returns:
        The resolved area.

    Raises:
        ValueError: If nothing was supplied, if a value cannot be parsed, if
            the bbox is inverted, or if the area is implausibly large.
        LookupError: If a place name does not match anything.
    """
    if bbox:
        parts = [part.strip() for part in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"--bbox needs four comma-separated numbers, got {bbox!r}"
            )
        try:
            west, south, east, north = (float(part) for part in parts)
        except ValueError as error:
            raise ValueError(f"--bbox has a non-numeric value: {bbox!r}") from error
        if west >= east or south >= north:
            raise ValueError(
                f"--bbox is inverted or empty: west must be < east and south "
                f"< north, got {bbox!r}"
            )
        resolved = Area(
            bbox=(west, south, east, north),
            center=((south + north) / 2.0, (west + east) / 2.0),
            label="custom bounding box",
        )

    elif center:
        parts = [part.strip() for part in center.split(",")]
        if len(parts) != 2:
            raise ValueError(
                f"--center needs 'lat,lon', got {center!r}"
            )
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError as error:
            raise ValueError(f"--center has a non-numeric value: {center!r}") from error
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"--center is out of range: {center!r}")
        resolved = Area(
            bbox=bbox_from_center(lat, lon, radius_mi),
            center=(lat, lon),
            label=f"{lat:.4f}, {lon:.4f}",
            radius_mi=radius_mi,
        )

    elif place:
        lat, lon, name = geocode(place)
        resolved = Area(
            bbox=bbox_from_center(lat, lon, radius_mi),
            center=(lat, lon),
            label=name,
            radius_mi=radius_mi,
        )

    else:
        raise ValueError(
            "give one of --area (a place name), --center lat,lon, or --bbox"
        )

    if resolved.area_sq_deg > MAX_AREA_SQ_DEG:
        raise ValueError(
            f"that area is {resolved.width_mi:.0f} x {resolved.height_mi:.0f} "
            "miles, which is too large to fetch in one go. Reduce --radius "
            "(8-12 miles is plenty for a marathon-length route) or pass a "
            "tighter --bbox."
        )
    if resolved.area_sq_deg > WARN_AREA_SQ_DEG:
        LOG.warning(
            "large area (%.0f x %.0f mi); the Overpass fetch may be slow",
            resolved.width_mi,
            resolved.height_mi,
        )

    LOG.info("area: %s", resolved.describe())
    return resolved


def looks_like_united_states(area: Area) -> bool:
    """Whether an area plausibly falls inside 3DEP's coverage.

    The elevation source is the USGS 3D Elevation Program, which covers the
    United States and its territories only. A course elsewhere needs a
    different DEM, so it is worth saying so before a download fails with
    something unhelpful.

    Returns:
        True if the area's centre sits inside a rough box around the US,
        including Alaska and Hawaii.
    """
    lat, lon = area.center
    boxes = (
        (24.0, 50.0, -125.5, -66.0),  # contiguous
        (51.0, 72.0, -170.0, -129.0),  # Alaska
        (18.0, 23.0, -161.0, -154.0),  # Hawaii
        (17.5, 18.6, -67.5, -65.2),  # Puerto Rico
    )
    return any(
        south <= lat <= north and west <= lon <= east
        for south, north, west, east in boxes
    )
