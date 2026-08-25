"""Coordinate projection for the Singapore case study: WGS84 -> UTM zone 48N.

Pure-Python transverse Mercator (Snyder, 1987) on WGS84. Singapore lies fully
inside UTM zone 48N (central meridian 105E), so zone-wide distortion is small
(scale factor ~1.0004 at the eastern edge) and all pipeline coordinates share
one consistent metric system: OSM nodes, GTFS stops and MATSim networks.

No third-party dependency (pyproj is intentionally avoided so the supply
pipeline stays deterministic and auditable).
"""
from __future__ import annotations

import math

# WGS84 ellipsoid
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)
_EP2 = _E2 / (1.0 - _E2)

# UTM zone 48N
_K0 = 0.9996
_CENTRAL_MERIDIAN = math.radians(105.0)
_FALSE_EASTING = 500000.0
_FALSE_NORTHING = 0.0


def _t(phi: float) -> float:
    return math.tan(phi) ** 2


def _c(phi: float) -> float:
    return _EP2 * math.cos(phi) ** 2


def utm48n(lat: float, lon: float) -> tuple[float, float]:
    """WGS84 (degrees) -> UTM 48N (easting, northing in meters).

    Standard transverse-Mercator series (Snyder pp. 61-64), adequate to
    sub-centimeter accuracy at Singapore latitudes.
    """
    phi = math.radians(lat)
    lam = math.radians(lon)
    dlam = lam - _CENTRAL_MERIDIAN

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    tan_phi = math.tan(phi)

    n = _A / math.sqrt(1.0 - _E2 * sin_phi * sin_phi)
    t = tan_phi * tan_phi
    c = _EP2 * cos_phi * cos_phi
    a_mer = cos_phi * dlam

    # meridian arc M from equator to phi
    e4 = _E2 * _E2
    e6 = e4 * _E2
    m = _A * (
        (1.0 - _E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * phi
        - (3.0 * _E2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * math.sin(2.0 * phi)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * math.sin(4.0 * phi)
        - (35.0 * e6 / 3072.0) * math.sin(6.0 * phi)
    )

    easting = _FALSE_EASTING + _K0 * n * (
        a_mer
        + (1.0 - t + c) * a_mer ** 3 / 6.0
        + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * _EP2) * a_mer ** 5 / 120.0
    )
    northing = _FALSE_NORTHING + _K0 * (
        m
        + n * tan_phi * (
            a_mer * a_mer / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c * c) * a_mer ** 4 / 24.0
            + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * _EP2) * a_mer ** 6 / 720.0
        )
    )
    return easting, northing


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters (independent check for the projection)."""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))
