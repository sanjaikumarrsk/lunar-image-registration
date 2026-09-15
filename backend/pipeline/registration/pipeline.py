"""End-to-end image-registration pipeline used by ``run_pipeline.py``."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from ...geometry.camera import CameraGeometryCapability, ChandrayaanGeometryGrid
from ..validation.dem import DEMProvider, GeoTiffDEMProvider, KaguyaDTMProvider, UnavailableDEMProvider, discover_dem_providers
from ..preprocessing.io import RasterSpec, compose_native_homography, native_to_preview, preview_raster, preview_to_native, robust_local_contrast
from ..matching.matching import (
    MatchSet,
    adaptive_ransac_threshold,
    adaptive_bidirectional_match,
    full_image_tiled_match,
    iterative_ransac_filter,
    multiscale_match,
    ransac_filter,
    registration_metrics,
    spatial_coverage,
    uniform_grid_select,
)
from ..refinement.refinement import refine_correspondence


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class InputContext:
    path: Path
    spec: RasterSpec
    native_spec: RasterSpec
    native_scale_from_input: tuple[float, float]
    geometry_path: Path | None
    label_path: Path | None
    product_id: str | None
    sensor: str | None
    geometry_grid: ChandrayaanGeometryGrid | None
    camera_capability: CameraGeometryCapability


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _derive_geometry(raw_path: Path) -> Path | None:
    text = raw_path.as_posix()
    if "/data/" in text:
        prefix, suffix = text.rsplit("/data/", 1)
        text = f"{prefix}/geometry/{suffix}"
    text = text.replace("_d_img_", "_g_grd_")
    candidate = ROOT / Path(text).with_suffix(".csv") if not Path(text).is_absolute() else Path(text).with_suffix(".csv")
    return candidate if candidate.exists() else None


def _catalog_for_path(path: Path) -> dict[str, str] | None:
    relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    catalog_path = ROOT / "data" / "chandrayaan_catalog.csv"
    if not catalog_path.exists():
        return None
    for row in _read_csv(catalog_path):
        if row.get("image_path") == relative or row.get("browse_path") == relative:
            return row
    return None


def _native_source_path(path: Path) -> tuple[Path, dict[str, str] | None]:
    catalog = _catalog_for_path(path)
    if catalog is not None:
        return ROOT / catalog["image_path"], catalog
    return path, None


def _camera_capability(path: Path, label_path: Path | None, geometry_path: Path | None, product_id: str | None, sensor: str | None) -> CameraGeometryCapability:
    if label_path is None or not label_path.exists():
        return CameraGeometryCapability(product_id or path.name, sensor or "unknown", geometry_path is not None, False, ("source_PDS4_label", "spacecraft_position", "attitude_reference_frame", "camera_principal_point", "lens_distortion_model", "camera_to_spacecraft_boresight_alignment"), "no complete source metadata")
    text = label_path.read_text(encoding="utf-8", errors="replace")
    path_text = path.as_posix()
    if "/data/" in path_text:
        prefix, suffix = path_text.rsplit("/data/", 1)
        path_text = f"{prefix}/miscellaneous/{suffix}"
    ancillary_oat = Path(path_text).with_suffix(".oat")
    ancillary_oath = ancillary_oat.with_suffix(".oath")
    oat_state_available = ancillary_oat.exists() and ancillary_oath.exists()
    missing = []
    for name, patterns in {
        "spacecraft_position": ("spacecraft.*position", "sub.*spacecraft"),
        "attitude_reference_frame": ("attitude.*reference", "reference.*frame"),
        "camera_principal_point": ("principal.*point", "optical.*axis"),
        "lens_distortion_model": ("distortion",),
        "camera_to_spacecraft_boresight_alignment": ("boresight", "camera.*alignment", "payload.*alignment"),
    }.items():
        if not any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            missing.append(name)
    if oat_state_available and "spacecraft_position" in missing:
        missing.remove("spacecraft_position")
    # OAT supplies state, but does not supply the missing payload calibration.
    reason = "OAT/OATH spacecraft state and PDS4 geolocation grid are available; payload camera calibration is incomplete" if oat_state_available else "PDS4 geolocation grid may be available, but spacecraft state and payload camera calibration are incomplete"
    return CameraGeometryCapability(product_id or path.name, sensor or "unknown", geometry_path is not None, False, tuple(missing), reason)


def build_input_context(path: str | Path, label_path: str | Path | None = None, load_geometry_grid: bool = False) -> InputContext:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    native_path, catalog = _native_source_path(path)
    explicit_label = Path(label_path) if label_path is not None else None
    if explicit_label is not None and not explicit_label.is_absolute():
        explicit_label = ROOT / explicit_label
    spec = RasterSpec.from_path(path, explicit_label)
    native_spec = RasterSpec.from_path(native_path, explicit_label if native_path == path else None)
    native_scale = (native_spec.shape[1] / spec.shape[1], native_spec.shape[0] / spec.shape[0])
    geometry_path = _derive_geometry(native_path)
    source_label = native_spec.label_path
    product_id = catalog.get("product_id") if catalog else None
    sensor = catalog.get("sensor") if catalog else None
    grid = None
    if load_geometry_grid and geometry_path is not None and source_label is not None and sensor is not None:
        grid = ChandrayaanGeometryGrid(geometry_path, source_label, product_id or native_path.name, sensor)
    capability = _camera_capability(native_path, source_label, geometry_path, product_id, sensor)
    return InputContext(path, spec, native_spec, native_scale, geometry_path, source_label, product_id, sensor, grid, capability)


def _provider_for_context(context: InputContext, explicit_dem: str | Path | None = None) -> DEMProvider:
    if explicit_dem is not None:
        path = Path(explicit_dem)
        if not path.is_absolute():
            path = ROOT / path
        if path.suffix.lower() in {".tif", ".tiff"}:
            return GeoTiffDEMProvider(path)
        if path.suffix.lower() == ".img":
            metadata_dir = path.parent.parent / "metadata"
            label_candidates = [
                metadata_dir / f"{path.stem}_dtm.lbl",
                metadata_dir / f"{path.stem.replace('_DTM', '_IMG')}_dtm.lbl",
            ]
            label = next((candidate for candidate in label_candidates if candidate.exists()), label_candidates[0])
            return KaguyaDTMProvider(path, label)
    providers = discover_dem_providers(ROOT)
    if context.path.parent.name == "tc_ortho":
        scene = context.path.stem
        for provider in providers:
            if isinstance(provider, KaguyaDTMProvider) and provider.data_path.stem.replace("_DTM", "_IMG") == scene:
                return provider
    name = context.path.name.lower()
    for provider in providers:
        if isinstance(provider, GeoTiffDEMProvider) and (("north" in name and "north" in provider.path.name) or ("south" in name and "south" in provider.path.name)):
            return provider
    return UnavailableDEMProvider("no_context_specific_dem")


def _source_geolocation(context: InputContext, point_preview: tuple[float, float], preview_scale: tuple[float, float]) -> tuple[float, float] | None:
    if context.geometry_path is None or context.label_path is None or context.sensor is None:
        return None
    if context.geometry_grid is None:
        context.geometry_grid = ChandrayaanGeometryGrid(context.geometry_path, context.label_path, context.product_id or context.native_spec.path.name, context.sensor)
    input_point = preview_to_native(point_preview, preview_scale[0], preview_scale[1])
    native_point = input_point * np.asarray(context.native_scale_from_input)
    return context.geometry_grid.pixel_to_latlon_nearest(float(native_point[0]), float(native_point[1]))


def _preview_to_original(context: InputContext, point_preview: tuple[float, float], preview_scale: tuple[float, float]) -> np.ndarray:
    """Convert a preview-pyramid point to the supplied raster's native grid."""

    input_point = preview_to_native(point_preview, preview_scale[0], preview_scale[1])
    return input_point * np.asarray(context.native_scale_from_input, dtype=np.float64)


def _native_to_working(context: InputContext, point_native: tuple[float, float] | np.ndarray, working_scale: tuple[float, float]) -> np.ndarray:
    """Convert a native point to the currently selected working image."""

    input_point = np.asarray(point_native, dtype=np.float64) / np.asarray(context.native_scale_from_input, dtype=np.float64)
    return native_to_preview(input_point, working_scale[0], working_scale[1])


def _source_geolocation_native(context: InputContext, point_native: tuple[float, float]) -> tuple[float, float] | None:
    """Look up source geolocation from a native pixel only when real geometry exists."""

    if context.geometry_path is None or context.label_path is None or context.sensor is None:
        return None
    if context.geometry_grid is None:
        context.geometry_grid = ChandrayaanGeometryGrid(
            context.geometry_path,
            context.label_path,
            context.product_id or context.native_spec.path.name,
            context.sensor,
        )
    return context.geometry_grid.pixel_to_latlon_nearest(float(point_native[0]), float(point_native[1]))


def _to_native_matches(matches: MatchSet, source_context: InputContext, reference_context: InputContext, source_scale: tuple[float, float], reference_scale: tuple[float, float]) -> MatchSet:
    source_points = np.asarray([
        _preview_to_original(source_context, (float(point[0]), float(point[1])), source_scale)
        for point in matches.source_points
    ], dtype=np.float32).reshape(-1, 2)
    reference_points = np.asarray([
        _preview_to_original(reference_context, (float(point[0]), float(point[1])), reference_scale)
        for point in matches.reference_points
    ], dtype=np.float32).reshape(-1, 2)
    return MatchSet(
        source_points,
        reference_points,
        matches.confidence.copy(),
        matches.method,
        matches.source_keypoints,
        matches.reference_keypoints,
        {**matches.metadata, "coordinate_space": "native_source_and_reference", "working_scale": {"source": list(source_scale), "reference": list(reference_scale)}},
    )


def _native_bounds_filter(matches: MatchSet, source_shape: tuple[int, int], reference_shape: tuple[int, int]) -> tuple[MatchSet, int]:
    source_height, source_width = source_shape
    reference_height, reference_width = reference_shape
    if not matches.count:
        return matches, 0
    points = matches.source_points
    ref_points = matches.reference_points
    keep = np.isfinite(points).all(axis=1) & np.isfinite(ref_points).all(axis=1)
    keep &= (points[:, 0] >= 0) & (points[:, 0] < source_width) & (points[:, 1] >= 0) & (points[:, 1] < source_height)
    keep &= (ref_points[:, 0] >= 0) & (ref_points[:, 0] < reference_width) & (ref_points[:, 1] >= 0) & (ref_points[:, 1] < reference_height)
    removed = int((~keep).sum())
    return MatchSet(
        points[keep],
        ref_points[keep],
        matches.confidence[keep],
        matches.method,
        matches.source_keypoints,
        matches.reference_keypoints,
        {**matches.metadata, "native_bounds_filtered": removed},
    ), removed


def _working_dimension_candidates(source_shape: tuple[int, int], reference_shape: tuple[int, int], max_dimension: int) -> list[int]:
    """Return the highest permitted working dimension without upscaling."""

    largest = max(max(source_shape), max(reference_shape))
    cap = max(1, min(768, int(max_dimension)))
    if largest <= cap:
        return [max(1, largest)]
    return [min(cap, largest)]


def _working_homography(native_homography: np.ndarray, source_context: InputContext, reference_context: InputContext, source_scale: tuple[float, float], reference_scale: tuple[float, float]) -> np.ndarray:
    source_native_to_working = np.asarray([[source_scale[0] / source_context.native_scale_from_input[0], 0.0, 0.0], [0.0, source_scale[1] / source_context.native_scale_from_input[1], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    reference_native_to_working = np.asarray([[reference_scale[0] / reference_context.native_scale_from_input[0], 0.0, 0.0], [0.0, reference_scale[1] / reference_context.native_scale_from_input[1], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return reference_native_to_working @ np.asarray(native_homography, dtype=np.float64) @ np.linalg.inv(source_native_to_working)


def _quality_sufficient(ransac: Any, matches: MatchSet) -> bool:
    if ransac.homography is None or ransac.inliers < 12:
        return False
    ratio = ransac.inliers / max(1, matches.count)
    errors = ransac.reprojection_errors_px[ransac.inlier_mask]
    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("inf")
    return bool(ratio >= 0.75 and rmse <= float(ransac.threshold_px))


def _write_match_image(source: np.ndarray, reference: np.ndarray, source_points: np.ndarray, reference_points: np.ndarray, mask: np.ndarray | None, path: Path) -> None:
    source_color = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    reference_color = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    keypoints0 = [cv2.KeyPoint(float(x), float(y), 3.0) for x, y in source_points]
    keypoints1 = [cv2.KeyPoint(float(x), float(y), 3.0) for x, y in reference_points]
    matches = [cv2.DMatch(index, index, 0.0) for index in range(len(source_points))]
    rendered = cv2.drawMatches(source_color, keypoints0, reference_color, keypoints1, matches, None, matchColor=(0, 220, 0), matchesMask=None if mask is None else mask.astype(np.uint8).tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(path), rendered)


def _as_display_gray(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.dtype == np.uint8:
        return array
    return cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _write_side_by_side(source: np.ndarray, reference: np.ndarray, path: Path) -> None:
    source8 = _as_display_gray(source)
    reference8 = _as_display_gray(reference)
    height = max(source8.shape[0], reference8.shape[0])
    canvas = np.zeros((height, source8.shape[1] + reference8.shape[1], 3), dtype=np.uint8)
    canvas[: source8.shape[0], : source8.shape[1]] = cv2.cvtColor(source8, cv2.COLOR_GRAY2BGR)
    canvas[: reference8.shape[0], source8.shape[1] :] = cv2.cvtColor(reference8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), canvas)


def _write_correspondence_visualization(source: np.ndarray, reference: np.ndarray, source_points: np.ndarray, reference_points: np.ndarray, inlier_mask: np.ndarray | None, path: Path, point_color: tuple[int, int, int] = (0, 220, 0)) -> None:
    """Draw point-to-point links on a side-by-side preview canvas."""

    source8 = _as_display_gray(source)
    reference8 = _as_display_gray(reference)
    height = max(source8.shape[0], reference8.shape[0])
    source_width = source8.shape[1]
    canvas = np.zeros((height, source_width + reference8.shape[1], 3), dtype=np.uint8)
    canvas[: source8.shape[0], :source_width] = cv2.cvtColor(source8, cv2.COLOR_GRAY2BGR)
    canvas[: reference8.shape[0], source_width:] = cv2.cvtColor(reference8, cv2.COLOR_GRAY2BGR)
    for index, (source_point, reference_point) in enumerate(zip(source_points, reference_points)):
        if inlier_mask is None:
            color = point_color
        else:
            color = (0, 220, 0) if bool(inlier_mask[index]) else (0, 0, 255)
        source_xy = (int(round(float(source_point[0]))), int(round(float(source_point[1]))))
        reference_xy = (source_width + int(round(float(reference_point[0]))), int(round(float(reference_point[1]))))
        cv2.line(canvas, source_xy, reference_xy, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, source_xy, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, reference_xy, 3, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _write_demo_difference(reference: np.ndarray, registered_source: np.ndarray | None, path: Path) -> None:
    reference8 = _as_display_gray(reference)
    if registered_source is None:
        difference = np.zeros_like(reference8)
    else:
        registered8 = _as_display_gray(registered_source)
        if registered8.shape != reference8.shape:
            registered8 = cv2.resize(registered8, (reference8.shape[1], reference8.shape[0]), interpolation=cv2.INTER_AREA)
        difference = cv2.absdiff(reference8, registered8)
    colored = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), colored)


def _write_uniform_csv(path: Path, matches: MatchSet, grid_diagnostic: dict[str, Any], source_context: InputContext, reference_context: InputContext, source_scale: tuple[float, float], reference_scale: tuple[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_x", "source_y", "reference_x", "reference_y", "confidence", "grid_row", "grid_column", "preview_source_x", "preview_source_y", "preview_reference_x", "preview_reference_y", "coordinate_space"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rows, columns = grid_diagnostic["grid"]
        height, width = reference_context.native_spec.shape
        for source, reference, confidence in zip(matches.source_points, matches.reference_points, matches.confidence):
            source_working = _native_to_working(source_context, source, source_scale)
            reference_working = _native_to_working(reference_context, reference, reference_scale)
            writer.writerow({"source_x": float(source[0]), "source_y": float(source[1]), "reference_x": float(reference[0]), "reference_y": float(reference[1]), "confidence": float(confidence), "grid_row": min(rows - 1, int(reference[1] / max(1, height) * rows)), "grid_column": min(columns - 1, int(reference[0] / max(1, width) * columns)), "preview_source_x": float(source_working[0]), "preview_source_y": float(source_working[1]), "preview_reference_x": float(reference_working[0]), "preview_reference_y": float(reference_working[1]), "coordinate_space": "native_source_and_reference"})


def _physical_validation(path: Path, context: InputContext, reference_context: InputContext, matches: MatchSet, inlier_mask: np.ndarray, dem: DEMProvider) -> int:
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(inlier_mask):
        source_native = matches.source_points[index]
        reference_native = matches.reference_points[index]
        geo = _source_geolocation_native(context, (float(source_native[0]), float(source_native[1])))
        elevation = None
        physical_available = False
        reason = "source_geolocation_unavailable"
        validity = "physical_validation_unavailable"
        if geo is not None:
            query = dem.get_elevation(*geo)
            if query.valid and query.elevation_m is not None:
                elevation = query.elevation_m
                reason = "real_dem_elevation_obtained_but_camera_projection_unavailable"
                validity = "dem_valid_physical_unavailable"
            else:
                reason = query.reason
                validity = "dem_unavailable"
        rows.append({"source_x": float(source_native[0]), "source_y": float(source_native[1]), "reference_x": float(reference_native[0]), "reference_y": float(reference_native[1]), "elevation": elevation, "physical_geometry_available": physical_available, "predicted_x": None, "predicted_y": None, "physical_residual": None, "validity": validity, "reason": reason})
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_x", "source_y", "reference_x", "reference_y", "elevation", "physical_geometry_available", "predicted_x", "predicted_y", "physical_residual", "validity", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _run_pipeline_legacy(source: str | Path, reference: str | Path, output_dir: str | Path, method: str = "loftr", max_dimension: int = 1024, grid: tuple[int, int] = (8, 8), max_per_cell: int = 4, source_label: str | Path | None = None, reference_label: str | Path | None = None, dem_path: str | Path | None = None, model_checkpoint: str | Path | None = None, synthetic_ground_truth: dict[str, Any] | None = None, confidence_threshold: float = 0.0) -> dict[str, Any]:
    legacy_start = time.perf_counter()
    output = Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    source_context = build_input_context(source, source_label, load_geometry_grid=False)
    reference_context = build_input_context(reference, reference_label, load_geometry_grid=False)
    source_preview, source_sx, source_sy = preview_raster(source_context.spec, max_dimension)
    reference_preview, reference_sx, reference_sy = preview_raster(reference_context.spec, max_dimension)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == "rift2":
        raise RuntimeError("RIFT2 is unavailable in this OpenCV build; no fallback was silently substituted")
    model = None
    if method == "loftr":
        from ..matching.matching import load_loftr
        checkpoint = Path(model_checkpoint) if model_checkpoint else ROOT / "models/finetuned_loftr/kaguya_geometric/best_model.pth"
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        model = load_loftr(checkpoint, device)
    matches = multiscale_match(method, source_preview, reference_preview, model, device, scales=(1.0, 0.5), confidence_threshold=confidence_threshold)
    uniform, grid_diagnostic = uniform_grid_select(matches, reference_preview.shape, grid, max_per_cell)
    ransac = ransac_filter(uniform)
    dem = _provider_for_context(source_context, dem_path)
    _write_match_image(source_preview, reference_preview, uniform.source_points, uniform.reference_points, None, output / "matches.png")
    _write_match_image(source_preview, reference_preview, uniform.source_points[ransac.inlier_mask], uniform.reference_points[ransac.inlier_mask], None, output / "inliers.png")
    is_demo_pair = source_context.path.parent.name.lower() == "demo" and reference_context.path.parent.name.lower() == "demo"
    if is_demo_pair:
        _write_side_by_side(source_context.spec.array(), reference_context.spec.array(), output / "source_reference.png")
        _write_correspondence_visualization(source_preview, reference_preview, uniform.source_points, uniform.reference_points, None, output / "candidate_matches.png", (255, 180, 0))
        _write_correspondence_visualization(source_preview, reference_preview, uniform.source_points, uniform.reference_points, ransac.inlier_mask, output / "ransac_inliers.png")
        _write_correspondence_visualization(source_preview, reference_preview, uniform.source_points[ransac.inlier_mask], uniform.reference_points[ransac.inlier_mask], None, output / "final_tie_points.png")
    _write_uniform_csv(output / "uniform_grid_points.csv", uniform, grid_diagnostic, source_context, reference_context, (source_sx, source_sy), (reference_sx, reference_sy))
    physical_count = _physical_validation(output / "physical_validation.csv", source_context, reference_context, uniform, ransac.inlier_mask, dem)
    homography = ransac.homography
    native_registration = {"used_native_source": False, "status": "not_registered"}
    native_h: np.ndarray | None = None
    registered_native: np.ndarray | None = None
    reference_native_for_output: np.ndarray | None = None
    if homography is not None:
        # Matching is performed in preview coordinates.  Convert directly
        # between native source/reference grids, including the case where a
        # catalog browse image was supplied while its native PDS product is
        # available underneath it.
        source_native_to_preview = (source_sx / source_context.native_scale_from_input[0], source_sy / source_context.native_scale_from_input[1])
        reference_native_to_preview = (reference_sx / reference_context.native_scale_from_input[0], reference_sy / reference_context.native_scale_from_input[1])
        native_h = compose_native_homography(homography, source_native_to_preview, reference_native_to_preview)
        try:
            source_native = source_context.native_spec.array()
            reference_native = reference_context.native_spec.array()
            warped = cv2.warpPerspective(source_native, native_h, (reference_native.shape[1], reference_native.shape[0]), flags=cv2.INTER_LINEAR)
            cv2.imwrite(str(output / "registered_source.png"), warped)
            ref8 = reference_native if reference_native.dtype == np.uint8 else cv2.normalize(reference_native, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            warp8 = warped if warped.dtype == np.uint8 else cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            overlay = cv2.addWeighted(cv2.cvtColor(ref8, cv2.COLOR_GRAY2BGR), 0.5, cv2.cvtColor(warp8, cv2.COLOR_GRAY2BGR), 0.5, 0.0)
            cv2.imwrite(str(output / "overlay.png"), overlay)
            registered_native = warped
            reference_native_for_output = reference_native
            native_registration = {"used_native_source": True, "status": "registered_with_native_source", "native_source_shape": list(source_native.shape), "native_reference_shape": list(reference_native.shape)}
        except (cv2.error, MemoryError) as error:
            native_registration = {"used_native_source": False, "status": "native_registration_failed", "reason": f"{type(error).__name__}:{error}"}
    if homography is None or not (output / "registered_source.png").exists():
        # Preserve a deterministic artifact even when RANSAC cannot produce a
        # transform.  When the caller supplied a browse image, retain that
        # supplied original rather than unexpectedly expanding it into a
        # multi-gigabyte native product.  This is explicitly an unregistered
        # source artifact, never presented as a successful registration.
        try:
            original = source_context.spec.array()
            cv2.imwrite(str(output / "registered_source.png"), original)
            registered_native = original
            reference_native_for_output = reference_context.spec.array()
            native_registration = {"used_native_source": False, "status": "unregistered_source_input_copy", "source_input_shape": list(original.shape)}
        except (cv2.error, MemoryError) as error:
            native_registration = {"used_native_source": False, "status": "native_source_artifact_failed", "reason": f"{type(error).__name__}:{error}"}
    if homography is not None:
        warped_preview = cv2.warpPerspective(cv2.cvtColor(source_preview, cv2.COLOR_GRAY2BGR), homography, (reference_preview.shape[1], reference_preview.shape[0]))
        overlay_preview = cv2.addWeighted(cv2.cvtColor(reference_preview, cv2.COLOR_GRAY2BGR), 0.5, warped_preview, 0.5, 0.0)
    else:
        overlay_preview = cv2.cvtColor(reference_preview, cv2.COLOR_GRAY2BGR)
    if not (output / "overlay.png").exists():
        # There is no registration overlay when RANSAC has no valid model;
        # write the reference preview as an explicit diagnostic artifact.
        cv2.imwrite(str(output / "overlay.png"), overlay_preview)
    if is_demo_pair:
        _write_demo_difference(reference_native_for_output if reference_native_for_output is not None else reference_context.spec.array(), registered_native, output / "difference.png")
    cv2.imwrite(str(output / "final_matching_result.png"), cv2.hconcat([cv2.cvtColor(reference_preview, cv2.COLOR_GRAY2BGR), overlay_preview]))
    if is_demo_pair and native_h is not None:
        _write_side_by_side(source_context.spec.array(), reference_context.spec.array(), output / "source_reference.png")
    tie_points: list[dict[str, Any]] = []
    native_reprojection_errors = np.full(uniform.count, np.nan, dtype=np.float64)
    if native_h is not None and uniform.count:
        native_source_points = np.asarray([_preview_to_original(source_context, (float(point[0]), float(point[1])), (source_sx, source_sy)) for point in uniform.source_points], dtype=np.float32)
        native_reference_points = np.asarray([_preview_to_original(reference_context, (float(point[0]), float(point[1])), (reference_sx, reference_sy)) for point in uniform.reference_points], dtype=np.float32)
        native_projected = cv2.perspectiveTransform(native_source_points.reshape(-1, 1, 2), native_h).reshape(-1, 2)
        native_reprojection_errors = np.linalg.norm(native_projected - native_reference_points, axis=1)
    for index in np.flatnonzero(ransac.inlier_mask):
        source_preview_point = tuple(float(value) for value in uniform.source_points[index])
        geo = _source_geolocation(source_context, source_preview_point, (source_sx, source_sy))
        dem_query = dem.get_elevation(*geo) if geo is not None else None
        result = refine_correspondence(source_preview_point, tuple(float(value) for value in uniform.reference_points[index]), float(uniform.confidence[index]), dem_query, SimpleNamespace(capability=source_context.camera_capability), source_image=source_preview, reference_image=reference_preview)
        row = dict(result.__dict__)
        row["subpixel_diagnostic"] = row["reprojection_error"]
        row["reprojection_error"] = None if not np.isfinite(native_reprojection_errors[index]) else float(native_reprojection_errors[index])
        row["match_distance"] = None
        row["inlier"] = True
        preview_values = {
            "preview_source_x": row["source_x"],
            "preview_source_y": row["source_y"],
            "preview_reference_x": row["reference_x"],
            "preview_reference_y": row["reference_y"],
            "preview_refined_source_x": row["refined_source_x"],
            "preview_refined_source_y": row["refined_source_y"],
            "preview_refined_reference_x": row["refined_reference_x"],
            "preview_refined_reference_y": row["refined_reference_y"],
        }
        source_original = _preview_to_original(source_context, (row["source_x"], row["source_y"]), (source_sx, source_sy))
        reference_original = _preview_to_original(reference_context, (row["reference_x"], row["reference_y"]), (reference_sx, reference_sy))
        refined_source_original = _preview_to_original(source_context, (row["refined_source_x"], row["refined_source_y"]), (source_sx, source_sy))
        refined_reference_original = _preview_to_original(reference_context, (row["refined_reference_x"], row["refined_reference_y"]), (reference_sx, reference_sy))
        row.update({
            "source_x": float(source_original[0]),
            "source_y": float(source_original[1]),
            "reference_x": float(reference_original[0]),
            "reference_y": float(reference_original[1]),
            "refined_source_x": float(refined_source_original[0]),
            "refined_source_y": float(refined_source_original[1]),
            "refined_reference_x": float(refined_reference_original[0]),
            "refined_reference_y": float(refined_reference_original[1]),
            "coordinate_space": "native_source_and_reference",
            **preview_values,
        })
        tie_points.append(row)
    with (output / "tie_points.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(tie_points[0].keys()) if tie_points else ["source_x", "source_y", "reference_x", "reference_y", "confidence", "match_distance", "reprojection_error", "inlier", "refined_source_x", "refined_source_y", "refined_reference_x", "refined_reference_y", "refinement_method", "physical_residual", "status", "reason", "physical_geometry_used", "coordinate_space", "subpixel_diagnostic", "preview_source_x", "preview_source_y", "preview_reference_x", "preview_reference_y", "preview_refined_source_x", "preview_refined_source_y", "preview_refined_reference_x", "preview_refined_reference_y"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tie_points)
    metrics = registration_metrics(uniform, ransac, reference_preview.shape)
    metrics.update({"method": method, "device": device.type, "source": str(source_context.path.relative_to(ROOT) if source_context.path.is_relative_to(ROOT) else source_context.path), "reference": str(reference_context.path.relative_to(ROOT) if reference_context.path.is_relative_to(ROOT) else reference_context.path), "original_source_dimensions": list(source_context.native_spec.shape), "original_reference_dimensions": list(reference_context.native_spec.shape), "working_source_dimensions": list(source_preview.shape), "working_reference_dimensions": list(reference_preview.shape), "working_scale_selected": max(source_preview.shape), "working_scale_attempts": [{"target_max_dimension": max(source_preview.shape), "quality_sufficient": homography is not None}], "raw_working_matches": matches.count, "native_candidate_matches_before_grid": matches.count, "grid_selected_matches": uniform.count, "loftr_runtime_seconds": None, "refinement_runtime_seconds": None, "total_pipeline_runtime_seconds": time.perf_counter() - legacy_start, "physical_geometry_available": False, "physical_reprojection_error": None, "physical_validation_rows": physical_count, "ground_truth_rmse_px": None, "registration_quality": "image_based_ransac_only" if homography is not None else "insufficient_matches", "native_registration": native_registration, "uniform_grid": grid_diagnostic, "camera_capability": source_context.camera_capability.__dict__, "coordinate_space": "native_source_and_reference", "preview_scale": {"source": [source_sx, source_sy], "reference": [reference_sx, reference_sy]}, "candidate_filtering": {"coordinate_bounds": True, "confidence_threshold": confidence_threshold, "uniform_grid_max_per_cell": max_per_cell, "ransac_threshold_px": ransac.threshold_px}, "match_distance_available": False if method == "loftr" else None, "validation": {"dem_provider": dem.name, "dem_elevation_validation": "unavailable" if isinstance(dem, UnavailableDEMProvider) else "queried_for_inliers", "physical_geometry": "unavailable"}, "output_artifacts": {"registered_source": "registered_source.png", "overlay": "overlay.png", "difference": "difference.png" if is_demo_pair else None, "tie_points": "tie_points.csv", "report": "report.md"}, "pipeline_stages": ["metadata_context_check", "catalog_localization", "dem_retrieval_validation", "multi_scale_illumination_robust_preprocessing", "loftr_or_baseline_matching", "candidate_filtering", "uniform_grid_selection", "image_ransac", "confidence_evaluation", "adaptive_subpixel_refinement", "dem_physical_validation_or_honest_fallback", "native_coordinate_tie_points", "registration_outputs_and_metrics"]})
    if synthetic_ground_truth is not None and homography is not None:
        metrics["ground_truth_rmse_px"] = _synthetic_rmse(uniform, synthetic_ground_truth)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    report = _make_report(metrics, source_context, reference_context, dem, output)
    (output / "report.md").write_text(report, encoding="utf-8")
    return metrics


def _ransac_rank(ransac: Any) -> tuple[int, float, float]:
    errors = ransac.reprojection_errors_px[ransac.inlier_mask]
    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("inf")
    ratio = ransac.inliers / max(1, len(ransac.inlier_mask))
    return ransac.inliers, ratio, -rmse


def _refine_ransac_inliers(
    matches: MatchSet,
    initial_ransac: Any,
    source_context: InputContext,
    reference_context: InputContext,
    source_image: np.ndarray,
    reference_image: np.ndarray,
    dem: DEMProvider,
    threshold_px: float,
) -> tuple[MatchSet, Any, dict[int, dict[str, Any]], dict[str, Any]]:
    """Refine only first-pass inliers, then re-estimate the final model."""

    refined_source = matches.source_points.copy()
    refined_reference = matches.reference_points.copy()
    records: dict[int, dict[str, Any]] = {}
    accepted = 0
    rejected = 0
    if initial_ransac.homography is None or not initial_ransac.inliers:
        return matches, initial_ransac, records, {"accepted": 0, "rejected": 0, "initial_inliers": initial_ransac.inliers, "final_inliers": initial_ransac.inliers}

    for index in np.flatnonzero(initial_ransac.inlier_mask):
        source_xy = tuple(float(value) for value in matches.source_points[index])
        reference_xy = tuple(float(value) for value in matches.reference_points[index])
        geo = _source_geolocation_native(source_context, source_xy)
        dem_query = dem.get_elevation(*geo) if geo is not None else None
        result = refine_correspondence(
            source_xy,
            reference_xy,
            float(matches.confidence[index]),
            dem_query,
            SimpleNamespace(capability=source_context.camera_capability),
            source_image=source_image,
            reference_image=reference_image,
        )
        candidate_source = np.asarray([[result.refined_source_x, result.refined_source_y]], dtype=np.float32)
        candidate_reference = np.asarray([[result.refined_reference_x, result.refined_reference_y]], dtype=np.float32)
        accept = False
        new_error = float("inf")
        if np.isfinite(candidate_source).all() and np.isfinite(candidate_reference).all():
            source_height, source_width = source_image.shape[:2]
            reference_height, reference_width = reference_image.shape[:2]
            accept = bool(
                0.0 <= candidate_source[0, 0] < source_width
                and 0.0 <= candidate_source[0, 1] < source_height
                and 0.0 <= candidate_reference[0, 0] < reference_width
                and 0.0 <= candidate_reference[0, 1] < reference_height
            )
            if accept:
                projected = cv2.perspectiveTransform(candidate_source.reshape(-1, 1, 2), initial_ransac.homography).reshape(-1, 2)
                new_error = float(np.linalg.norm(projected[0] - candidate_reference[0]))
                old_error = float(initial_ransac.reprojection_errors_px[index])
                accept = bool(np.isfinite(new_error) and new_error <= old_error + 1e-6)
        if accept:
            refined_source[index] = candidate_source[0]
            refined_reference[index] = candidate_reference[0]
            accepted += 1
        else:
            rejected += 1
        records[int(index)] = {"result": result, "accepted": accept, "new_error": new_error}

    refined_matches = MatchSet(
        refined_source,
        refined_reference,
        matches.confidence.copy(),
        matches.method,
        matches.source_keypoints,
        matches.reference_keypoints,
        {**matches.metadata, "subpixel_refined_matches": accepted, "subpixel_rejected_matches": rejected},
    )
    # Refit from all accepted first-pass inliers, then run a second robust
    # estimate restricted to that accepted set.  The old homography is never
    # used as the final model after coordinates have changed.
    refit_homography, _ = cv2.findHomography(
        refined_source[initial_ransac.inlier_mask],
        refined_reference[initial_ransac.inlier_mask],
        0,
    )
    if refit_homography is not None:
        projected = cv2.perspectiveTransform(refined_source.reshape(-1, 1, 2), refit_homography).reshape(-1, 2)
        refit_errors = np.linalg.norm(projected - refined_reference, axis=1)
        refit_mask = initial_ransac.inlier_mask & np.isfinite(refit_errors) & (refit_errors <= threshold_px)
        refit_ransac = type(initial_ransac)(refit_homography, refit_mask, refit_errors, threshold_px)
    else:
        refit_ransac = type(initial_ransac)(None, np.zeros(matches.count, dtype=bool), np.full(matches.count, np.nan), threshold_px)
    final_ransac = iterative_ransac_filter(refined_matches, threshold_px, initial_ransac.inlier_mask)
    candidates = [candidate for candidate in (refit_ransac, final_ransac) if candidate.homography is not None]
    chosen = max(candidates, key=_ransac_rank) if candidates else final_ransac
    return refined_matches, chosen, records, {
        "accepted": accepted,
        "rejected": rejected,
        "initial_inliers": initial_ransac.inliers,
        "second_pass_inliers": final_ransac.inliers,
        "final_inliers": chosen.inliers,
        "model_recomputed": bool(refit_ransac.homography is not None or final_ransac.homography is not None),
    }


def run_pipeline(source: str | Path, reference: str | Path, output_dir: str | Path, method: str = "loftr", max_dimension: int = 768, grid: tuple[int, int] = (8, 8), max_per_cell: int = 0, source_label: str | Path | None = None, reference_label: str | Path | None = None, dem_path: str | Path | None = None, model_checkpoint: str | Path | None = None, synthetic_ground_truth: dict[str, Any] | None = None, confidence_threshold: float = 0.0, pipeline_mode: str = "improved") -> dict[str, Any]:
    """Run registration with a bounded LoFTR working image and native outputs.

    The only images passed to LoFTR are the selected working previews.  Every
    returned match is immediately converted to native coordinates; spatial
    selection, RANSAC, refinement, DEM lookup, and registration then operate
    in the native coordinate system.
    """

    if pipeline_mode == "legacy":
        return _run_pipeline_legacy(
            source,
            reference,
            output_dir,
            method,
            max_dimension,
            grid,
            4 if max_per_cell <= 0 else max_per_cell,
            source_label,
            reference_label,
            dem_path,
            model_checkpoint,
            synthetic_ground_truth,
            confidence_threshold,
        )
    if pipeline_mode != "improved":
        raise ValueError(f"Unknown pipeline mode: {pipeline_mode}")
    total_start = time.perf_counter()
    output = Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    source_context = build_input_context(source, source_label, load_geometry_grid=False)
    reference_context = build_input_context(reference, reference_label, load_geometry_grid=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == "rift2":
        raise RuntimeError("RIFT2 is unavailable in this OpenCV build; no fallback was silently substituted")

    model = None
    if method == "loftr":
        from ..matching.matching import load_loftr
        checkpoint = Path(model_checkpoint) if model_checkpoint else ROOT / "models/finetuned_loftr/kaguya_geometric/best_model.pth"
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        model = load_loftr(checkpoint, device)

    targets = _working_dimension_candidates(source_context.spec.shape, reference_context.spec.shape, max_dimension)
    working_attempts: list[dict[str, Any]] = []
    matching_runtime = 0.0
    selected: tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float], MatchSet, MatchSet, dict[str, Any], Any] | None = None
    selected_rank: tuple[int, float, float] | None = None
    for attempt_number, target in enumerate(targets, start=1):
        source_work, source_sx, source_sy = preview_raster(source_context.spec, target)
        reference_work, reference_sx, reference_sy = preview_raster(reference_context.spec, target)
        source_work = robust_local_contrast(source_work)
        reference_work = robust_local_contrast(reference_work)
        match_start = time.perf_counter()
        # LoFTR sees only this bounded, illumination-robust working view.  A
        # reverse pass is used for mutual consistency; native images are not
        # passed to the matcher.
        effective_confidence_threshold = max(float(confidence_threshold), 0.25) if method == "loftr" else float(confidence_threshold)
        working_matches = adaptive_bidirectional_match(
            method,
            source_work,
            reference_work,
            model,
            device,
            confidence_threshold=effective_confidence_threshold,
            mutual_radius_px=8.0,
            dedup_radius_px=3.0,
        )
        # A single full-frame LoFTR view can over-sample its strongest local
        # texture.  Run overlapping tiles over the entire frame as well, with
        # each reference tile inferred from the coarse full-frame model.
        coverage_seed = ransac_filter(working_matches, threshold_px=2.0)
        working_matches = full_image_tiled_match(
            method,
            source_work,
            reference_work,
            model,
            device,
            working_matches,
            coverage_seed.homography,
            confidence_threshold=effective_confidence_threshold,
            mutual_radius_px=8.0,
            dedup_radius_px=3.0,
        )
        match_runtime = time.perf_counter() - match_start
        matching_runtime += match_runtime
        native_candidates, bounds_removed = _native_bounds_filter(
            _to_native_matches(working_matches, source_context, reference_context, (source_sx, source_sy), (reference_sx, reference_sy)),
            source_context.native_spec.shape,
            reference_context.native_spec.shape,
        )
        uniform_attempt, grid_attempt = uniform_grid_select(native_candidates, reference_context.native_spec.shape, grid, max_per_cell)
        ransac_threshold, threshold_diagnostic = adaptive_ransac_threshold((source_sx, source_sy), (reference_sx, reference_sy), working_noise_px=1.5)
        ransac_attempt = iterative_ransac_filter(uniform_attempt, ransac_threshold)
        quality = _quality_sufficient(ransac_attempt, uniform_attempt)
        working_attempts.append({
            "attempt": attempt_number,
            "target_max_dimension": target,
            "source_dimensions": list(source_work.shape),
            "reference_dimensions": list(reference_work.shape),
            "source_scale_xy": [source_sx, source_sy],
            "reference_scale_xy": [reference_sx, reference_sy],
            "raw_matches": working_matches.count,
            "forward_raw_matches": working_matches.metadata.get("forward_raw_matches"),
            "reverse_raw_matches": working_matches.metadata.get("reverse_raw_matches"),
            "mutual_matches_before_filtering": working_matches.metadata.get("mutual_matches_before_filtering"),
            "internal_scales": working_matches.metadata.get("internal_scales"),
            "internal_fallback_used": working_matches.metadata.get("internal_fallback_used"),
            "low_confidence_threshold": effective_confidence_threshold,
            "native_bounds_removed": bounds_removed,
            "candidate_matches": native_candidates.count,
            "uniform_matches": uniform_attempt.count,
            "ransac_inliers": ransac_attempt.inliers,
            "ransac_outliers": ransac_attempt.outliers,
            "inlier_ratio": ransac_attempt.inliers / max(1, uniform_attempt.count),
            "ransac_threshold_px": ransac_threshold,
            "ransac_threshold_diagnostic": threshold_diagnostic,
            "quality_sufficient": quality,
            "matching_runtime_seconds": match_runtime,
            "coverage_tiles_attempted": working_matches.metadata.get("coverage_tiles_attempted"),
            "coverage_tiles_with_matches": working_matches.metadata.get("coverage_tiles_with_matches"),
            "valid_overlap_reference_bbox_working": working_matches.metadata.get("valid_overlap_reference_bbox_working"),
        })
        attempt_rank = _ransac_rank(ransac_attempt)
        if selected is None or selected_rank is None or attempt_rank > selected_rank:
            selected = (source_work, reference_work, (source_sx, source_sy), (reference_sx, reference_sy), working_matches, uniform_attempt, grid_attempt, ransac_attempt)
            selected_rank = attempt_rank
        if quality or attempt_number == len(targets):
            break

    if selected is None:
        raise RuntimeError("No working-scale matching attempt was executed")
    source_work, reference_work, source_scale, reference_scale, working_matches, uniform, grid_diagnostic, ransac = selected
    dem = _provider_for_context(source_context, dem_path)
    is_demo_pair = source_context.path.parent.name.lower() == "demo" and reference_context.path.parent.name.lower() == "demo"
    before_refinement_metrics = registration_metrics(uniform, ransac, reference_context.native_spec.shape)
    largest_before_refinement = [
        {
            "match_index": int(index),
            "source_x": float(uniform.source_points[index, 0]),
            "source_y": float(uniform.source_points[index, 1]),
            "reference_x": float(uniform.reference_points[index, 0]),
            "reference_y": float(uniform.reference_points[index, 1]),
            "error_px": float(ransac.reprojection_errors_px[index]),
        }
        for index in sorted(np.flatnonzero(ransac.inlier_mask), key=lambda index: float(ransac.reprojection_errors_px[index]), reverse=True)[:10]
    ]

    # Refinement is deliberately restricted to first-pass RANSAC inliers and
    # uses native-resolution patches.  The accepted coordinates are then fed
    # back into a second robust model estimate before registration.
    refinement_start = time.perf_counter()
    source_native_image: np.ndarray | None = source_context.native_spec.array() if ransac.inliers else None
    reference_native_image: np.ndarray | None = reference_context.native_spec.array() if ransac.inliers else None
    if source_native_image is not None and reference_native_image is not None:
        uniform, ransac, refinement_records, refinement_info = _refine_ransac_inliers(
            uniform,
            ransac,
            source_context,
            reference_context,
            source_native_image,
            reference_native_image,
            dem,
            ransac.threshold_px,
        )
    else:
        refinement_records = {}
        refinement_info = {"accepted": 0, "rejected": 0, "initial_inliers": ransac.inliers, "second_pass_inliers": ransac.inliers, "final_inliers": ransac.inliers}
    refinement_runtime = time.perf_counter() - refinement_start
    selected_attempt = next(
        (
            attempt
            for attempt in working_attempts
            if attempt["source_dimensions"] == list(source_work.shape)
            and attempt["reference_dimensions"] == list(reference_work.shape)
        ),
        working_attempts[-1],
    )
    homography = ransac.homography
    working_h = _working_homography(homography, source_context, reference_context, source_scale, reference_scale) if homography is not None else None

    # Diagnostics are deliberately rendered in the working coordinate system;
    # the CSV and RANSAC values below remain native coordinates.
    uniform_source_work = np.asarray([_native_to_working(source_context, point, source_scale) for point in uniform.source_points], dtype=np.float32).reshape(-1, 2)
    uniform_reference_work = np.asarray([_native_to_working(reference_context, point, reference_scale) for point in uniform.reference_points], dtype=np.float32).reshape(-1, 2)
    _write_match_image(source_work, reference_work, uniform_source_work, uniform_reference_work, None, output / "matches.png")
    _write_match_image(source_work, reference_work, uniform_source_work[ransac.inlier_mask], uniform_reference_work[ransac.inlier_mask], None, output / "inliers.png")
    if is_demo_pair:
        _write_side_by_side(source_context.spec.array(), reference_context.spec.array(), output / "source_reference.png")
        _write_correspondence_visualization(source_work, reference_work, uniform_source_work, uniform_reference_work, None, output / "candidate_matches.png", (255, 180, 0))
        _write_correspondence_visualization(source_work, reference_work, uniform_source_work, uniform_reference_work, ransac.inlier_mask, output / "ransac_inliers.png")
        _write_correspondence_visualization(source_work, reference_work, uniform_source_work[ransac.inlier_mask], uniform_reference_work[ransac.inlier_mask], None, output / "final_tie_points.png")
    # Always render final correspondences in native full-image coordinates so
    # coverage can be checked independently of the working preview.
    if source_native_image is not None and reference_native_image is not None:
        _write_correspondence_visualization(
            source_native_image,
            reference_native_image,
            uniform.source_points[ransac.inlier_mask],
            uniform.reference_points[ransac.inlier_mask],
            None,
            output / "full_image_tie_points.png",
        )
    _write_uniform_csv(output / "uniform_grid_points.csv", uniform, grid_diagnostic, source_context, reference_context, source_scale, reference_scale)
    physical_count = _physical_validation(output / "physical_validation.csv", source_context, reference_context, uniform, ransac.inlier_mask, dem)

    native_registration = {"used_native_source": False, "status": "not_registered"}
    registered_native: np.ndarray | None = None
    reference_native_for_output: np.ndarray | None = None
    native_h: np.ndarray | None = homography
    if native_h is not None:
        try:
            source_native = source_context.native_spec.array()
            reference_native = reference_context.native_spec.array()
            warped = cv2.warpPerspective(source_native, native_h, (reference_native.shape[1], reference_native.shape[0]), flags=cv2.INTER_LINEAR)
            cv2.imwrite(str(output / "registered_source.png"), warped)
            ref8 = _as_display_gray(reference_native)
            warp8 = _as_display_gray(warped)
            overlay = cv2.addWeighted(cv2.cvtColor(ref8, cv2.COLOR_GRAY2BGR), 0.5, cv2.cvtColor(warp8, cv2.COLOR_GRAY2BGR), 0.5, 0.0)
            cv2.imwrite(str(output / "overlay.png"), overlay)
            registered_native = warped
            reference_native_for_output = reference_native
            native_registration = {"used_native_source": True, "status": "registered_with_native_source", "native_source_shape": list(source_native.shape), "native_reference_shape": list(reference_native.shape)}
        except (cv2.error, MemoryError) as error:
            native_registration = {"used_native_source": False, "status": "native_registration_failed", "reason": f"{type(error).__name__}:{error}"}
    if native_h is None or not (output / "registered_source.png").exists():
        try:
            original = source_context.spec.array()
            cv2.imwrite(str(output / "registered_source.png"), original)
            registered_native = original
            reference_native_for_output = reference_context.spec.array()
            native_registration = {"used_native_source": False, "status": "unregistered_source_input_copy", "source_input_shape": list(original.shape)}
        except (cv2.error, MemoryError) as error:
            native_registration = {"used_native_source": False, "status": "native_source_artifact_failed", "reason": f"{type(error).__name__}:{error}"}

    if working_h is not None:
        warped_work = cv2.warpPerspective(cv2.cvtColor(source_work, cv2.COLOR_GRAY2BGR), working_h, (reference_work.shape[1], reference_work.shape[0]))
        overlay_work = cv2.addWeighted(cv2.cvtColor(reference_work, cv2.COLOR_GRAY2BGR), 0.5, warped_work, 0.5, 0.0)
    else:
        overlay_work = cv2.cvtColor(reference_work, cv2.COLOR_GRAY2BGR)
    if not (output / "overlay.png").exists():
        cv2.imwrite(str(output / "overlay.png"), overlay_work)
    if is_demo_pair:
        _write_demo_difference(reference_native_for_output if reference_native_for_output is not None else reference_context.spec.array(), registered_native, output / "difference.png")
    cv2.imwrite(str(output / "final_matching_result.png"), cv2.hconcat([cv2.cvtColor(reference_work, cv2.COLOR_GRAY2BGR), overlay_work]))

    tie_points: list[dict[str, Any]] = []
    for index in np.flatnonzero(ransac.inlier_mask):
        record = refinement_records.get(int(index))
        if record is None:
            continue
        result = record["result"]
        row = dict(result.__dict__)
        row["subpixel_diagnostic"] = row["reprojection_error"]
        row["reprojection_error"] = None if not np.isfinite(ransac.reprojection_errors_px[index]) else float(ransac.reprojection_errors_px[index])
        row["match_distance"] = None
        row["inlier"] = True
        if not record["accepted"]:
            row["refined_source_x"] = row["source_x"]
            row["refined_source_y"] = row["source_y"]
            row["refined_reference_x"] = row["reference_x"]
            row["refined_reference_y"] = row["reference_y"]
            row["reason"] = f"{row['reason']};refinement_rejected_geometric_error_increased"
        source_native_point = (float(row["source_x"]), float(row["source_y"]))
        reference_native_point = (float(row["reference_x"]), float(row["reference_y"]))
        source_working = _native_to_working(source_context, source_native_point, source_scale)
        reference_working = _native_to_working(reference_context, reference_native_point, reference_scale)
        refined_source_working = _native_to_working(source_context, (row["refined_source_x"], row["refined_source_y"]), source_scale)
        refined_reference_working = _native_to_working(reference_context, (row["refined_reference_x"], row["refined_reference_y"]), reference_scale)
        row.update({
            "coordinate_space": "native_source_and_reference",
            "preview_source_x": float(source_working[0]),
            "preview_source_y": float(source_working[1]),
            "preview_reference_x": float(reference_working[0]),
            "preview_reference_y": float(reference_working[1]),
            "preview_refined_source_x": float(refined_source_working[0]),
            "preview_refined_source_y": float(refined_source_working[1]),
            "preview_refined_reference_x": float(refined_reference_working[0]),
            "preview_refined_reference_y": float(refined_reference_working[1]),
        })
        tie_points.append(row)
    with (output / "tie_points.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(tie_points[0].keys()) if tie_points else ["source_x", "source_y", "reference_x", "reference_y", "confidence", "match_distance", "reprojection_error", "inlier", "refined_source_x", "refined_source_y", "refined_reference_x", "refined_reference_y", "refinement_method", "physical_residual", "status", "reason", "physical_geometry_used", "coordinate_space", "subpixel_diagnostic", "preview_source_x", "preview_source_y", "preview_reference_x", "preview_reference_y", "preview_refined_source_x", "preview_refined_source_y", "preview_refined_reference_x", "preview_refined_reference_y"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tie_points)

    metrics = registration_metrics(uniform, ransac, reference_context.native_spec.shape)
    metrics["source_tie_point_spatial_coverage"] = spatial_coverage(uniform.source_points[ransac.inlier_mask], source_context.native_spec.shape)
    metrics["refinement_comparison"] = {
        "before": before_refinement_metrics,
        "after": dict(metrics),
        "largest_inlier_errors_before_refinement": largest_before_refinement,
    }
    metrics.update({
        "method": method,
        "device": "CUDA" if device.type == "cuda" else "CPU",
        "source": str(source_context.path.relative_to(ROOT) if source_context.path.is_relative_to(ROOT) else source_context.path),
        "reference": str(reference_context.path.relative_to(ROOT) if reference_context.path.is_relative_to(ROOT) else reference_context.path),
        "original_source_dimensions": list(source_context.native_spec.shape),
        "original_reference_dimensions": list(reference_context.native_spec.shape),
        "working_source_dimensions": list(source_work.shape),
        "working_reference_dimensions": list(reference_work.shape),
        "working_scale_factor": {
            "source_native_to_working_xy": [source_scale[0] / source_context.native_scale_from_input[0], source_scale[1] / source_context.native_scale_from_input[1]],
            "reference_native_to_working_xy": [reference_scale[0] / reference_context.native_scale_from_input[0], reference_scale[1] / reference_context.native_scale_from_input[1]],
        },
        "working_scale_target_max_dimension": max(source_work.shape[0], source_work.shape[1], reference_work.shape[0], reference_work.shape[1]),
        "working_scale_attempts": working_attempts,
        "working_scale_selected": selected_attempt["target_max_dimension"],
        "full_image_matching": {
            "enabled": bool(working_matches.metadata.get("coverage_full_image", False)),
            "tile_grid": working_matches.metadata.get("coverage_tile_grid"),
            "tile_overlap": working_matches.metadata.get("coverage_tile_overlap"),
            "tiles_attempted": working_matches.metadata.get("coverage_tiles_attempted"),
            "tiles_with_matches": working_matches.metadata.get("coverage_tiles_with_matches"),
            "valid_overlap_reference_bbox_working": working_matches.metadata.get("valid_overlap_reference_bbox_working"),
        },
        "raw_working_matches": selected_attempt["raw_matches"],
        "raw_forward_matches": selected_attempt.get("forward_raw_matches"),
        "raw_reverse_matches": selected_attempt.get("reverse_raw_matches"),
        "mutual_matches": selected_attempt.get("mutual_matches_before_filtering"),
        "native_candidate_matches_before_grid": selected_attempt["candidate_matches"],
        "grid_selected_matches": uniform.count,
        "preview_scale": {"source": list(source_scale), "reference": list(reference_scale)},
        "loftr_runtime_seconds": matching_runtime if method == "loftr" else None,
        "refinement_runtime_seconds": refinement_runtime,
        "physical_geometry_available": False,
        "physical_reprojection_error": None,
        "physical_validation_rows": physical_count,
        "ground_truth_rmse_px": None,
        "registration_quality": "image_based_ransac_only" if native_h is not None else "insufficient_matches",
        "native_registration": native_registration,
        "uniform_grid": grid_diagnostic,
        "subpixel_refinement": refinement_info,
        "camera_capability": source_context.camera_capability.__dict__,
        "coordinate_space": "native_source_and_reference",
        "candidate_filtering": {
            "coordinate_bounds": True,
            "confidence_threshold": effective_confidence_threshold,
            "mutual_consistency": True,
            "mutual_radius_px_working": 8.0,
            "near_duplicate_radius_px_working": 3.0,
            "uniform_grid_max_per_cell": max_per_cell,
            "ransac_threshold_px": ransac.threshold_px,
        },
        "match_distance_available": False if method == "loftr" else None,
        "validation": {"dem_provider": dem.name, "dem_elevation_validation": "unavailable" if isinstance(dem, UnavailableDEMProvider) else "queried_for_inliers", "physical_geometry": "unavailable"},
         "output_artifacts": {"registered_source": "registered_source.png", "overlay": "overlay.png", "difference": "difference.png" if is_demo_pair else None, "tie_points": "tie_points.csv", "full_image_tie_points": "full_image_tie_points.png", "report": "report.md"},
        "pipeline_stages": ["metadata_context_check", "catalog_localization", "dem_retrieval_validation", "multi_scale_illumination_robust_preprocessing", "working_scale_selection", "bidirectional_loftr_matching", "confidence_mutual_and_duplicate_filtering", "native_coordinate_conversion", "uniform_grid_selection", "adaptive_image_ransac", "iterative_model_refinement", "confidence_evaluation", "accepted_inlier_subpixel_refinement", "second_robust_estimation", "dem_physical_validation_or_honest_fallback", "native_coordinate_tie_points", "original_resolution_registration", "registration_outputs_and_metrics"],
    })
    if synthetic_ground_truth is not None and native_h is not None:
        metrics["ground_truth_rmse_px"] = _synthetic_rmse(uniform, synthetic_ground_truth)
    metrics["total_pipeline_runtime_seconds"] = time.perf_counter() - total_start
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    report = _make_report(metrics, source_context, reference_context, dem, output)
    (output / "report.md").write_text(report, encoding="utf-8")
    return metrics


def _synthetic_rmse(matches: MatchSet, truth: dict[str, Any]) -> float | None:
    homography = np.asarray(truth["homography"], dtype=np.float32)
    if matches.count == 0:
        return None
    projected = cv2.perspectiveTransform(matches.source_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.linalg.norm(projected - matches.reference_points, axis=1) ** 2)))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _make_report(metrics: dict[str, Any], source: InputContext, reference: InputContext, dem: DEMProvider, output: Path) -> str:
    return f"""# Registration run

Method: `{metrics['method']}`  
Source: `{source.path}`  
Reference: `{reference.path}`  

## Metrics

- Original source/reference dimensions: `{metrics['original_source_dimensions']}` / `{metrics['original_reference_dimensions']}`
- LoFTR working source/reference dimensions: `{metrics['working_source_dimensions']}` / `{metrics['working_reference_dimensions']}`
- Selected working target: `{metrics['working_scale_selected']}` px; attempts: `{len(metrics['working_scale_attempts'])}`
- Device: `{metrics['device']}`; LoFTR runtime: `{metrics['loftr_runtime_seconds']}` s; refinement runtime: `{metrics['refinement_runtime_seconds']}` s
- Total pipeline runtime: `{metrics['total_pipeline_runtime_seconds']}` s
- Working/raw candidates: {metrics['raw_working_matches']} / native candidates before grid: {metrics['native_candidate_matches_before_grid']}
- Grid-selected candidates: {metrics['grid_selected_matches']}
- RANSAC inliers/outliers: {metrics['ransac_inliers']}/{metrics['ransac_outliers']}
- Inlier ratio: {metrics['inlier_ratio']:.4f}
- Mean/median/max/RMSE RANSAC error: {metrics['mean_reprojection_error_px']}, {metrics['median_reprojection_error_px']}, {metrics['maximum_reprojection_error_px']}, {metrics['rmse_reprojection_error_px']} px
- Mean/minimum candidate confidence: {metrics['mean_match_confidence']}, {metrics['minimum_match_confidence']}
- GT RMSE: {metrics['ground_truth_rmse_px']} (null for real imagery without ground truth)
- Physical residual: null; exact physical camera projection is unavailable

## Pipeline stages

Executed: metadata/context check, catalog localization, multi-scale preprocessing, illumination-robust representation, automatic working-scale selection, `{metrics['method']}` matching, coordinate/confidence filtering, native-coordinate conversion, uniform-grid selection, native-coordinate RANSAC, confidence evaluation, adaptive classical sub-pixel refinement on accepted inliers, native DEM/physical validation or honest fallback, original-resolution registration, and output metrics.

DEM/physical stage: `{metrics['validation']['dem_elevation_validation']}` DEM provider `{metrics['validation']['dem_provider']}`; physical camera validation unavailable and therefore not used. No ground-truth correspondences were supplied for this pair. LoFTR descriptor match distance is unavailable; confidence is reported instead.

Final tie points: `{metrics['ransac_inliers']}` in `{metrics['coordinate_space']}`. Registration status: `{metrics['native_registration']['status']}`.

## Geometry and DEM

- DEM provider: `{dem.name}`
- Physical geometry available: `false`
- Fallback: image-based matching, RANSAC, and classical local subpixel diagnostic
- Final tie-point coordinate space: `native_source_and_reference`
- Camera limitation: `{source.camera_capability.reason}`

The homography in this run is an image-based RANSAC registration model only; it is not used as physical camera geometry.
"""
