"""Small geodesy helpers shared by the analysis and reporting steps.

Distances use the haversine formula on a spherical earth. Over the few
kilometres any single segment covers, the error against an ellipsoidal
calculation is well under a metre, which is far below the DEM's own accuracy.
"""

from __future__ import annotations

import math

import numpy as np

#: Mean earth radius in metres (IUGG).
EARTH_RADIUS_M = 6_371_008.8

#: Metres per statute mile.
METRES_PER_MILE = 1609.344


def haversine_m(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance between two points in metres.

    Args:
        lat1: Latitude of the first point, degrees.
        lon1: Longitude of the first point, degrees.
        lat2: Latitude of the second point, degrees.
        lon2: Longitude of the second point, degrees.

    Returns:
        Distance in metres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def cumulative_distance_m(coords: np.ndarray) -> np.ndarray:
    """Cumulative along-path distance for a polyline.

    Args:
        coords: ``(n, 2)`` array of ``(longitude, latitude)`` in degrees.

    Returns:
        ``(n,)`` array of metres from the first vertex; the first element is 0.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return np.zeros(coords.shape[0], dtype=float)

    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    d_lat = np.diff(lat)
    d_lon = np.diff(lon)
    a = (
        np.sin(d_lat / 2.0) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(d_lon / 2.0) ** 2
    )
    steps = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample_polyline(
    coords: np.ndarray, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to evenly spaced points along its length.

    Args:
        coords: ``(n, 2)`` array of ``(longitude, latitude)`` in degrees.
        spacing_m: Desired spacing between output points, metres.

    Returns:
        A tuple of ``(points, distances)`` where ``points`` is an ``(m, 2)``
        array of interpolated ``(longitude, latitude)`` pairs and
        ``distances`` is the ``(m,)`` array of along-path metres. The first
        and last vertices of the input are always present. A polyline shorter
        than ``spacing_m`` is returned as just its endpoints.

        The actual spacing is the largest value not exceeding ``spacing_m``
        that divides the path into whole steps, so the result is never
        sparser than requested.

    Raises:
        ValueError: If ``spacing_m`` is not positive.
    """
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")

    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return coords.copy(), np.zeros(coords.shape[0], dtype=float)

    along = cumulative_distance_m(coords)
    total = float(along[-1])
    if total <= 0.0:
        return coords[:1].copy(), np.zeros(1, dtype=float)

    count = max(2, int(math.ceil(total / spacing_m)) + 1)
    targets = np.linspace(0.0, total, count)
    lon = np.interp(targets, along, coords[:, 0])
    lat = np.interp(targets, along, coords[:, 1])
    return np.column_stack([lon, lat]), targets


def bearing_deg(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Initial great-circle bearing from the first point to the second.

    Args:
        lat1: Latitude of the origin, degrees.
        lon1: Longitude of the origin, degrees.
        lat2: Latitude of the destination, degrees.
        lon2: Longitude of the destination, degrees.

    Returns:
        Bearing in degrees clockwise from north, in ``[0, 360)``.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(
        phi2
    ) * math.cos(d_lambda)
    return math.degrees(math.atan2(y, x)) % 360.0


def turn_angle_deg(bearing_in: float, bearing_out: float) -> float:
    """Absolute change of direction between two bearings.

    Args:
        bearing_in: Bearing arriving at the junction, degrees.
        bearing_out: Bearing leaving the junction, degrees.

    Returns:
        The turn magnitude in degrees, in ``[0, 180]``. Zero means dead
        straight ahead; 180 means a full reversal.
    """
    diff = (bearing_out - bearing_in + 180.0) % 360.0 - 180.0
    return abs(diff)


def metres_to_miles(metres: float) -> float:
    """Convert metres to statute miles."""
    return metres / METRES_PER_MILE


def miles_to_metres(miles: float) -> float:
    """Convert statute miles to metres."""
    return miles * METRES_PER_MILE


def google_maps_link(lat: float, lon: float) -> str:
    """A Google Maps URL that drops a pin on the given point."""
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"
