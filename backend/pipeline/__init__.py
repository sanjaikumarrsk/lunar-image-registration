"""Reusable lunar geometry and image-registration pipeline components."""

from ..geometry.camera import CameraGeometryCapability, ChandrayaanGeometryGrid
from ..geometry.orbit_attitude import OATRecord, OATHMetadata, OrbitAttitudeSeries, parse_oat_record, parse_oath
from ..geometry.pds import PolarStereographic, parse_pds3_label
from ..geometry.physical_geometry import CameraIntrinsics, CameraPose, PhysicalCameraModel, PhysicalGeometryCapability, PhysicalGeometryUnavailable
from .refinement.refinement import refine_correspondence, select_refinement_branch
from .validation.dem import DEMProvider, DEMQuery, GeoTiffDEMProvider, KaguyaDTMProvider, UnavailableDEMProvider, discover_dem_providers
from .validation.physics import LunarReferenceFrame, terrain_point_from_dem, terrain_point_from_latlon

__all__ = [
    "CameraGeometryCapability", "ChandrayaanGeometryGrid", "DEMProvider", "DEMQuery",
    "GeoTiffDEMProvider", "KaguyaDTMProvider", "LunarReferenceFrame", "PolarStereographic",
    "UnavailableDEMProvider", "discover_dem_providers", "parse_pds3_label", "refine_correspondence",
    "select_refinement_branch", "terrain_point_from_dem", "terrain_point_from_latlon", "OATRecord",
    "OATHMetadata", "OrbitAttitudeSeries", "parse_oat_record", "parse_oath", "CameraIntrinsics",
    "CameraPose", "PhysicalCameraModel", "PhysicalGeometryCapability", "PhysicalGeometryUnavailable",
]
