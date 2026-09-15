"""Camera-geometry capability checks without inventing a camera model."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraGeometryCapability:
    product_id: str
    sensor: str
    map_grid_available: bool
    exact_physical_projection_available: bool
    missing_information: tuple[str, ...]
    reason: str


class ChandrayaanGeometryGrid:
    """Native-image pixel to selenographic lookup from the PDS4 grid CSV."""

    def __init__(self, csv_path: str | Path, source_label_path: str | Path, product_id: str, sensor: str) -> None:
        self.csv_path = Path(csv_path)
        self.source_label_path = Path(source_label_path)
        self.product_id = product_id
        self.sensor = sensor
        rows = list(csv.DictReader(self.csv_path.open(newline="", encoding="utf-8")))
        if not rows:
            raise ValueError(f"Empty geometry grid: {csv_path}")
        self.points = np.asarray([[float(row["Pixel"]), float(row["Scan"])] for row in rows], dtype=np.float64)
        self.longitude = np.asarray([float(row["Longitude"]) for row in rows], dtype=np.float64)
        self.latitude = np.asarray([float(row["Latitude"]) for row in rows], dtype=np.float64)
        self.capability = self._inspect_capability()

    def _inspect_capability(self) -> CameraGeometryCapability:
        text = self.source_label_path.read_text(encoding="utf-8", errors="replace")
        present = {key for key in ("focal_length", "spacecraft_altitude", "roll", "pitch", "yaw") if re.search(key, text, re.IGNORECASE)}
        missing = tuple(name for name in ("spacecraft_position_latitude", "spacecraft_position_longitude", "attitude_reference_frame", "camera_principal_point", "lens_distortion_model") if name not in present)
        return CameraGeometryCapability(self.product_id, self.sensor, True, False, missing, "PDS4 geolocation grid exists, but a complete sensor projection model is unavailable")

    def pixel_to_latlon_nearest(self, x: float, y: float) -> tuple[float, float]:
        distances = np.sum((self.points - np.asarray([x, y])) ** 2, axis=1)
        index = int(np.argmin(distances))
        return float(self.latitude[index]), float(self.longitude[index])

    def project_terrain_point(self, xyz_m: tuple[float, float, float]) -> tuple[float, float]:
        raise RuntimeError("exact physical camera projection unavailable: missing spacecraft position, frame-attached attitude, principal point, and distortion model")

