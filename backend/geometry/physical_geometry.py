"""Explicit physical camera geometry primitives.

The classes here implement real pinhole/radial-tangential projection when a
validated camera-to-lunar pose and calibration are supplied.  They do not
silently substitute image homographies, map grids, zero distortion, or a
center-of-image principal point for missing metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..pipeline.validation.physics import LunarReferenceFrame, terrain_point_from_dem


class PhysicalGeometryUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class PhysicalGeometryCapability:
    available: bool
    missing_information: tuple[str, ...]
    source_files: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CameraIntrinsics:
    focal_length_m: float
    pixel_size_m: tuple[float, float]
    image_size_px: tuple[int, int]
    principal_point_px: tuple[float, float] | None
    distortion_coefficients: tuple[float, ...] | None
    boresight_camera_to_body: np.ndarray | None

    @property
    def focal_pixels(self) -> tuple[float, float]:
        return (self.focal_length_m / self.pixel_size_m[0], self.focal_length_m / self.pixel_size_m[1])

    def capability(self) -> PhysicalGeometryCapability:
        missing: list[str] = []
        if self.principal_point_px is None:
            missing.append("camera_principal_point")
        if self.distortion_coefficients is None:
            missing.append("lens_distortion_parameters")
        if self.boresight_camera_to_body is None:
            missing.append("camera_to_spacecraft_boresight_alignment")
        return PhysicalGeometryCapability(not missing, tuple(missing), (), "camera intrinsics/alignment are incomplete" if missing else "complete camera calibration")


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in the body-fixed lunar Cartesian frame.

    ``camera_to_lunar`` maps camera-frame vectors to lunar-frame vectors.
    ``position_m`` is the camera center in the same lunar frame.
    """

    position_m: np.ndarray
    camera_to_lunar: np.ndarray
    frame_name: str = "lunar_body_fixed_planetocentric"

    def validate(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64).reshape(3)
        rotation = np.asarray(self.camera_to_lunar, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("camera pose contains non-finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("camera_to_lunar is not a proper rotation matrix")


def _radial_tangential_distort(x: float, y: float, coefficients: tuple[float, ...]) -> tuple[float, float]:
    values = list(coefficients) + [0.0] * max(0, 5 - len(coefficients))
    k1, k2, p1, p2, k3 = values[:5]
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
    return x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x), y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y


@dataclass
class PhysicalCameraModel:
    intrinsics: CameraIntrinsics
    pose: CameraPose
    frame: LunarReferenceFrame = LunarReferenceFrame()

    def __post_init__(self) -> None:
        self.pose.validate()
        capability = self.capability()
        if not capability.available:
            raise PhysicalGeometryUnavailable(capability.reason + ": " + ", ".join(capability.missing_information))

    def capability(self) -> PhysicalGeometryCapability:
        camera_capability = self.intrinsics.capability()
        pose_missing: list[str] = []
        try:
            self.pose.validate()
        except ValueError as error:
            pose_missing.append(str(error))
        missing = tuple(camera_capability.missing_information) + tuple(pose_missing)
        return PhysicalGeometryCapability(not missing, missing, camera_capability.source_files, "complete physical camera model" if not missing else "physical camera model unavailable")

    def world_to_camera(self, point_m: Iterable[float]) -> np.ndarray:
        return self.pose.camera_to_lunar.T @ (np.asarray(tuple(point_m), dtype=np.float64) - self.pose.position_m)

    def project_point(self, point_m: Iterable[float]) -> tuple[float, float]:
        camera = self.world_to_camera(point_m)
        if camera[2] <= 0.0:
            raise PhysicalGeometryUnavailable("terrain point is behind the camera")
        x = float(camera[0] / camera[2])
        y = float(camera[1] / camera[2])
        xd, yd = _radial_tangential_distort(x, y, self.intrinsics.distortion_coefficients or ())
        fx, fy = self.intrinsics.focal_pixels
        cx, cy = self.intrinsics.principal_point_px or (float("nan"), float("nan"))
        pixel = (fx * xd + cx, fy * yd + cy)
        if not all(math.isfinite(value) for value in pixel):
            raise PhysicalGeometryUnavailable("projection produced a non-finite pixel")
        return pixel

    def pixel_to_ray(self, pixel_xy: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        if self.intrinsics.principal_point_px is None or self.intrinsics.distortion_coefficients is None:
            raise PhysicalGeometryUnavailable("inverse projection requires principal point and distortion parameters")
        fx, fy = self.intrinsics.focal_pixels
        cx, cy = self.intrinsics.principal_point_px
        xd = (float(tuple(pixel_xy)[0]) - cx) / fx
        yd = (float(tuple(pixel_xy)[1]) - cy) / fy
        x, y = xd, yd
        for _ in range(12):
            predicted_x, predicted_y = _radial_tangential_distort(x, y, self.intrinsics.distortion_coefficients)
            x += xd - predicted_x
            y += yd - predicted_y
        direction_camera = np.asarray([x, y, 1.0], dtype=np.float64)
        direction_camera /= np.linalg.norm(direction_camera)
        direction_lunar = self.pose.camera_to_lunar @ direction_camera
        direction_lunar /= np.linalg.norm(direction_lunar)
        return self.pose.position_m.copy(), direction_lunar

    def project_latlon_elevation(self, latitude: float, longitude: float, elevation_m: float) -> tuple[float, float]:
        return self.project_point(self.frame.to_cartesian(latitude, longitude, elevation_m))

    def intersect_pixel_with_dem(self, pixel_xy: Iterable[float], dem: Any, max_iterations: int = 12) -> dict[str, Any]:
        origin, direction = self.pixel_to_ray(pixel_xy)
        radius = self.frame.radius_m
        point = _ray_sphere_intersection(origin, direction, radius)
        if point is None:
            return {"status": "unavailable", "reason": "camera_ray_misses_lunar_reference_sphere", "point_m": None}
        for _ in range(max_iterations):
            latitude, longitude, _ = self.frame.from_cartesian(tuple(point))
            query = dem.get_elevation(latitude, longitude)
            if not query.valid or query.elevation_m is None:
                return {"status": "unavailable", "reason": query.reason, "point_m": None, "latitude": latitude, "longitude": longitude}
            next_point = _ray_sphere_intersection(origin, direction, radius + query.elevation_m)
            if next_point is None:
                return {"status": "unavailable", "reason": "ray_misses_dem_surface", "point_m": None}
            if float(np.linalg.norm(next_point - point)) < 1e-4:
                point = next_point
                break
            point = next_point
        latitude, longitude, elevation = self.frame.from_cartesian(tuple(point))
        return {"status": "valid", "reason": "dem_intersection", "point_m": tuple(float(value) for value in point), "latitude": latitude, "longitude": longitude, "elevation_m": elevation, "real_elevation": True}


def _ray_sphere_intersection(origin: np.ndarray, direction: np.ndarray, radius_m: float) -> np.ndarray | None:
    b = 2.0 * float(np.dot(origin, direction))
    c = float(np.dot(origin, origin) - radius_m * radius_m)
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return None
    roots = [(-b - math.sqrt(discriminant)) / 2.0, (-b + math.sqrt(discriminant)) / 2.0]
    positive = [root for root in roots if root > 0.0]
    return origin + min(positive) * direction if positive else None


def unavailable_capability(source_files: Iterable[str], missing_information: Iterable[str], reason: str) -> PhysicalGeometryCapability:
    return PhysicalGeometryCapability(False, tuple(missing_information), tuple(source_files), reason)
