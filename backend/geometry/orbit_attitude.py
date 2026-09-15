"""Parser for Chandrayaan-2 ORBTATTD orbit/attitude ancillary files.

The local ISRO ancillary readme defines fixed-width ASCII OAT records.  This
module keeps the supplied frames and units explicit and never converts an
unknown quaternion convention into a camera pose implicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


def _float(line: str, start: int, end: int) -> float:
    return float(line[start:end].strip())


def _ints(line: str, start: int, end: int) -> list[int]:
    text = line[start:end]
    return [int(text[index : index + 4].strip()) for index in range(0, len(text), 4) if text[index : index + 4].strip()]


def _quat(line: str, start: int) -> tuple[float, float, float, float]:
    return tuple(_float(line, start + index * 14, start + (index + 1) * 14) for index in range(4))  # type: ignore[return-value]


def _datetime_from_record(line: str) -> datetime:
    year, month, day, hour, minute, second, millis = _ints(line, 18, 46)
    return datetime(year, month, day, hour, minute, second, millis * 1000, tzinfo=timezone.utc)


@dataclass(frozen=True)
class OATRecord:
    record_number: int
    time: datetime
    lunar_position_j2000_km: tuple[float, float, float]
    spacecraft_position_lunar_j2000_km: tuple[float, float, float]
    spacecraft_velocity_km_s: tuple[float, float, float]
    inertial_to_body_quaternion_xyzw: tuple[float, float, float, float]
    inertial_to_earth_fixed_quaternion_xyzw: tuple[float, float, float, float]
    inertial_to_lunar_fixed_quaternion_xyzw: tuple[float, float, float, float]
    subspacecraft_latitude_deg: float
    subspacecraft_longitude_deg: float
    solar_azimuth_deg: float
    solar_elevation_deg: float
    footprint_latitude_deg: float
    footprint_longitude_deg: float
    spacecraft_altitude_km: float
    emission_angle_deg: float
    slant_range_km: float
    orbit_number: int
    payload_fov_velocity_angle_deg: float
    yaw_deg: float
    roll_deg: float
    pitch_deg: float

    @property
    def spacecraft_position_m(self) -> np.ndarray:
        return np.asarray(self.spacecraft_position_lunar_j2000_km, dtype=np.float64) * 1000.0


@dataclass(frozen=True)
class OATHMetadata:
    source_attitude: int | None
    mission_phase: int | None
    record_count: int | None
    record_length: int | None
    start_time: datetime | None
    end_time: datetime | None


def parse_oath(path: str | Path) -> OATHMetadata:
    data = Path(path).read_bytes()
    line = data.splitlines()[0].decode("ascii", errors="replace")
    def integer(start: int, end: int) -> int | None:
        try:
            return int(line[start:end].strip())
        except ValueError:
            return None
    def time(start: int) -> datetime | None:
        try:
            values = _ints(line, start, start + 28)
            if len(values) != 7:
                return None
            return datetime(values[0], values[1], values[2], values[3], values[4], values[5], values[6] * 1000, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            return None
    return OATHMetadata(
        source_attitude=integer(111, 112),
        mission_phase=integer(112, 113),
        record_count=integer(99, 105),
        record_length=integer(105, 111),
        start_time=time(43),
        end_time=time(71),
    )


def parse_oat_record(line: bytes | str) -> OATRecord:
    text = line.decode("ascii", errors="replace") if isinstance(line, bytes) else line
    if len(text.rstrip("\r\n")) < 587 or not text.startswith("ORBTATTD"):
        raise ValueError("not a complete ORBTATTD record")
    text = text.rstrip("\r\n")
    return OATRecord(
        record_number=int(text[8:14].strip()),
        time=_datetime_from_record(text),
        lunar_position_j2000_km=(_float(text, 46, 66), _float(text, 66, 86), _float(text, 86, 106)),
        spacecraft_position_lunar_j2000_km=(_float(text, 106, 126), _float(text, 126, 146), _float(text, 146, 166)),
        spacecraft_velocity_km_s=(_float(text, 166, 178), _float(text, 178, 190), _float(text, 190, 202)),
        inertial_to_body_quaternion_xyzw=_quat(text, 202),
        inertial_to_earth_fixed_quaternion_xyzw=_quat(text, 258),
        inertial_to_lunar_fixed_quaternion_xyzw=_quat(text, 314),
        subspacecraft_latitude_deg=_float(text, 370, 384),
        subspacecraft_longitude_deg=_float(text, 384, 398),
        solar_azimuth_deg=_float(text, 398, 412),
        solar_elevation_deg=_float(text, 412, 426),
        footprint_latitude_deg=_float(text, 426, 440),
        footprint_longitude_deg=_float(text, 440, 454),
        spacecraft_altitude_km=_float(text, 454, 466),
        emission_angle_deg=_float(text, 479, 488),
        slant_range_km=_float(text, 506, 516),
        orbit_number=int(text[516:521].strip()),
        payload_fov_velocity_angle_deg=_float(text, 530, 539),
        yaw_deg=_float(text, 539, 555),
        roll_deg=_float(text, 555, 571),
        pitch_deg=_float(text, 571, 587),
    )


@dataclass
class OrbitAttitudeSeries:
    records: list[OATRecord]
    oath: OATHMetadata | None = None

    @classmethod
    def from_files(cls, oat_path: str | Path, oath_path: str | Path | None = None) -> "OrbitAttitudeSeries":
        records = [parse_oat_record(line) for line in Path(oat_path).read_bytes().splitlines() if line.startswith(b"ORBTATTD")]
        if not records:
            raise ValueError(f"No ORBTATTD records in {oat_path}")
        return cls(records, parse_oath(oath_path) if oath_path is not None and Path(oath_path).exists() else None)

    @property
    def start_time(self) -> datetime:
        return self.records[0].time

    @property
    def end_time(self) -> datetime:
        return self.records[-1].time

    def nearest(self, time: datetime) -> OATRecord:
        return min(self.records, key=lambda record: abs((record.time - time).total_seconds()))

    def validate(self) -> dict[str, object]:
        position_norms = [float(np.linalg.norm(record.spacecraft_position_m)) for record in self.records]
        quaternion_norms = [float(np.linalg.norm(record.inertial_to_body_quaternion_xyzw)) for record in self.records]
        return {
            "records": len(self.records),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "position_radius_km": {"minimum": min(position_norms) / 1000.0, "maximum": max(position_norms) / 1000.0},
            "body_quaternion_norm": {"minimum": min(quaternion_norms), "maximum": max(quaternion_norms)},
            "mission_phase": self.oath.mission_phase if self.oath else None,
            "source_attitude": self.oath.source_attitude if self.oath else None,
        }


def quat_xyzw_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """Convert an explicitly scalar-last quaternion to a 3x3 rotation matrix."""

    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
