"""PDS3 label parsing and verified polar-stereographic transformations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


def parse_pds3_label(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="ascii", errors="replace")
    result: dict[str, Any] = {"_path": str(path), "_text": text}
    for line in text.splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        key, value = match.groups()
        value = value.split("<", 1)[0].strip()
        if value.startswith('"') and value.endswith('"'):
            result[key] = value[1:-1]
        else:
            number = re.match(rf"^({_NUMBER})$", value)
            if number:
                result[key] = float(number.group(1))
            else:
                result[key] = value
    return result


def _number(label: dict[str, Any], name: str) -> float:
    value = label.get(name)
    if value is None:
        raise KeyError(f"PDS label field {name!r} is missing")
    return float(value)


def _text(label: dict[str, Any], name: str) -> str:
    value = label.get(name)
    if value is None:
        raise KeyError(f"PDS label field {name!r} is missing")
    return str(value).strip().strip('"')


def normalize_longitude(longitude_deg: float) -> float:
    value = float(longitude_deg) % 360.0
    return value if value >= 0.0 else value + 360.0


def unwrap_longitude(longitude_deg: float, reference_deg: float) -> float:
    value = float(longitude_deg)
    while value - reference_deg > 180.0:
        value -= 360.0
    while value - reference_deg < -180.0:
        value += 360.0
    return value


@dataclass(frozen=True)
class PolarStereographic:
    """SELENE/Kaguya PDS polar stereographic map definition.

    Pixel coordinates are zero-based array coordinates measured at pixel
    centers. PDS line/sample coordinates are one-based, so the verified
    offset equations are:

    ``x_km = (sample_zero_based - SAMPLE_PROJECTION_OFFSET) * MAP_SCALE``
    ``y_km = (LINE_PROJECTION_OFFSET - line_zero_based) * MAP_SCALE``
    """

    hemisphere: int
    center_longitude_deg: float
    radius_km: float
    map_scale_km_per_pixel: float
    line_projection_offset: float
    sample_projection_offset: float
    lines: int
    samples: int
    projection: str = "STEREOGRAPHIC"
    coordinate_system: str = "PLANETOCENTRIC"

    @classmethod
    def from_label(cls, label: dict[str, Any]) -> "PolarStereographic":
        projection = _text(label, "MAP_PROJECTION_TYPE").upper()
        coordinate_system = _text(label, "COORDINATE_SYSTEM_NAME").upper()
        center_latitude = _number(label, "CENTER_LATITUDE")
        if projection != "STEREOGRAPHIC":
            raise ValueError(f"Unsupported projection: {projection}")
        if coordinate_system != "PLANETOCENTRIC":
            raise ValueError(f"Unsupported coordinate system: {coordinate_system}")
        if abs(abs(center_latitude) - 90.0) > 1e-8:
            raise ValueError("The supported stereographic definition is polar")
        return cls(
            hemisphere=1 if center_latitude > 0 else -1,
            center_longitude_deg=_number(label, "CENTER_LONGITUDE"),
            radius_km=_number(label, "A_AXIS_RADIUS"),
            map_scale_km_per_pixel=_number(label, "MAP_SCALE"),
            line_projection_offset=_number(label, "LINE_PROJECTION_OFFSET"),
            sample_projection_offset=_number(label, "SAMPLE_PROJECTION_OFFSET"),
            lines=int(_number(label, "LINE_LAST_PIXEL")),
            samples=int(_number(label, "SAMPLE_LAST_PIXEL")),
            projection=projection,
            coordinate_system=coordinate_system,
        )

    def pixel_to_projected_km(self, x: float, y: float) -> tuple[float, float]:
        return (
            (float(x) - self.sample_projection_offset) * self.map_scale_km_per_pixel,
            (self.line_projection_offset - float(y)) * self.map_scale_km_per_pixel,
        )

    def projected_km_to_pixel(self, x_km: float, y_km: float) -> tuple[float, float]:
        return (
            float(x_km) / self.map_scale_km_per_pixel + self.sample_projection_offset,
            self.line_projection_offset - float(y_km) / self.map_scale_km_per_pixel,
        )

    def pixel_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        px, py = self.pixel_to_projected_km(x, y)
        rho = math.hypot(px, py)
        central_angle = 2.0 * math.atan(rho / (2.0 * self.radius_km))
        latitude = self.hemisphere * (90.0 - math.degrees(central_angle))
        if rho < 1e-15:
            longitude = self.center_longitude_deg
        else:
            theta = math.atan2(px, -self.hemisphere * py)
            longitude = normalize_longitude(self.center_longitude_deg + math.degrees(theta))
        return latitude, longitude

    def latlon_to_pixel(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
        latitude = float(latitude_deg)
        if latitude * self.hemisphere <= 0.0 or abs(latitude) > 90.0:
            raise ValueError("Latitude is outside this polar hemisphere")
        colatitude = math.radians(90.0 - abs(latitude))
        rho = 2.0 * self.radius_km * math.tan(colatitude / 2.0)
        theta = math.radians(unwrap_longitude(longitude_deg, self.center_longitude_deg) - self.center_longitude_deg)
        px = rho * math.sin(theta)
        py = -self.hemisphere * rho * math.cos(theta)
        return self.projected_km_to_pixel(px, py)

    def contains_pixel(self, x: float, y: float) -> bool:
        return 0.0 <= float(x) < self.samples and 0.0 <= float(y) < self.lines

    def round_trip_pixel(self, x: float, y: float) -> tuple[float, float, float]:
        lat, lon = self.pixel_to_latlon(x, y)
        rx, ry = self.latlon_to_pixel(lat, lon)
        return rx, ry, math.hypot(rx - x, ry - y)

    def round_trip_latlon(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float, float]:
        x, y = self.latlon_to_pixel(latitude_deg, longitude_deg)
        rlat, rlon = self.pixel_to_latlon(x, y)
        dlon = math.radians(unwrap_longitude(rlon, longitude_deg) - longitude_deg)
        dlat = math.radians(rlat - latitude_deg)
        return rlat, rlon, math.hypot(dlat, dlon)
