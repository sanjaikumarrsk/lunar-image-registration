"""Lunar reference-frame and terrain-point construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LunarReferenceFrame:
    """Planetocentric, body-fixed lunar Cartesian frame.

    +X is latitude 0, longitude 0; +Y is longitude 90E; +Z is north. The
    radius is the mean lunar radius used by the Kaguya PDS labels. Elevation
    is added to that radius in metres. No elevation is supplied by default.
    """

    radius_m: float = 1_737_400.0
    longitude_direction: str = "EAST"
    latitude_type: str = "PLANETOCENTRIC"

    def to_cartesian(self, latitude_deg: float, longitude_deg: float, elevation_m: float) -> tuple[float, float, float]:
        radius = self.radius_m + float(elevation_m)
        if radius <= 0.0:
            raise ValueError("Terrain radius must be positive")
        lat = math.radians(float(latitude_deg))
        lon = math.radians(float(longitude_deg))
        cos_lat = math.cos(lat)
        return (
            radius * cos_lat * math.cos(lon),
            radius * cos_lat * math.sin(lon),
            radius * math.sin(lat),
        )

    def from_cartesian(self, xyz_m: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = (float(value) for value in xyz_m)
        radius = math.sqrt(x * x + y * y + z * z)
        if radius <= 0.0:
            raise ValueError("Cartesian point has zero radius")
        latitude = math.degrees(math.asin(z / radius))
        longitude = math.degrees(math.atan2(y, x)) % 360.0
        return latitude, longitude, radius - self.radius_m


def terrain_point_from_latlon(
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float | None,
    frame: LunarReferenceFrame | None = None,
) -> tuple[float, float, float]:
    if elevation_m is None:
        raise ValueError("A real DEM elevation is required; no default elevation is allowed")
    return (frame or LunarReferenceFrame()).to_cartesian(latitude_deg, longitude_deg, float(elevation_m))


def terrain_point_from_dem(
    dem: Any,
    latitude_deg: float,
    longitude_deg: float,
    frame: LunarReferenceFrame | None = None,
) -> tuple[tuple[float, float, float] | None, dict[str, Any]]:
    query = dem.get_elevation(latitude_deg, longitude_deg)
    if not query.valid or query.elevation_m is None:
        return None, {"status": "unavailable", "reason": query.reason, "real_elevation": False}
    point = terrain_point_from_latlon(latitude_deg, longitude_deg, query.elevation_m, frame)
    return point, {
        "status": "valid",
        "reason": query.reason,
        "elevation_m": query.elevation_m,
        "quality": query.quality,
        "real_elevation": True,
    }
