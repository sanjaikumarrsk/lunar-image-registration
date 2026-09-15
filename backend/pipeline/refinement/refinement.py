"""Hybrid classical/physical correspondence refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class RefinementResult:
    source_x: float
    source_y: float
    reference_x: float
    reference_y: float
    refined_source_x: float
    refined_source_y: float
    refined_reference_x: float
    refined_reference_y: float
    refinement_method: str
    reprojection_error: float | None
    physical_residual: float | None
    confidence: float
    status: str
    reason: str
    physical_geometry_used: bool

    @property
    def residual_px(self) -> float | None:
        return self.reprojection_error


def select_refinement_branch(confidence: float, dem_valid: bool, physical_geometry_available: bool, threshold: float = 0.75) -> str:
    if confidence >= threshold:
        return "classical"
    if dem_valid and physical_geometry_available:
        return "dem_physical"
    return "classical_fallback"


def _patch(image: np.ndarray, point: tuple[float, float], size: int) -> np.ndarray | None:
    half = size // 2
    x, y = point
    if x < half or y < half or x >= image.shape[1] - half or y >= image.shape[0] - half:
        return None
    # Extract only the local source window.  Converting a whole native PDS
    # raster to float32 here would defeat the working-scale optimization.
    x0 = max(0, int(math.floor(x)) - half - 1)
    y0 = max(0, int(math.floor(y)) - half - 1)
    x1 = min(image.shape[1], int(math.floor(x)) + half + 2)
    y1 = min(image.shape[0], int(math.floor(y)) + half + 2)
    local = image[y0:y1, x0:x1]
    if local.size == 0:
        return None
    center = (float(x - x0), float(y - y0))
    return cv2.getRectSubPix(local, (size, size), center).astype(np.float32, copy=False)


def local_subpixel_refine(source_image: np.ndarray, reference_image: np.ndarray, source_xy: tuple[float, float], reference_xy: tuple[float, float], window: int = 17) -> tuple[tuple[float, float], tuple[float, float], float | None, str]:
    """Refine a match using local phase correlation.

    The residual is a local correlation diagnostic, not ground-truth error.
    """

    size = max(9, int(window) | 1)
    source_patch = _patch(source_image, source_xy, size)
    reference_patch = _patch(reference_image, reference_xy, size)
    if source_patch is None or reference_patch is None:
        return source_xy, reference_xy, None, "boundary_or_invalid_patch"
    source_patch = source_patch - float(source_patch.mean())
    reference_patch = reference_patch - float(reference_patch.mean())
    if float(np.std(source_patch)) < 1e-6 or float(np.std(reference_patch)) < 1e-6:
        return source_xy, reference_xy, None, "low_texture_patch"
    try:
        shift, response = cv2.phaseCorrelate(source_patch, reference_patch)
    except cv2.error:
        return source_xy, reference_xy, None, "phase_correlation_failed"
    refined_reference = (float(reference_xy[0] + shift[0]), float(reference_xy[1] + shift[1]))
    diagnostic = float(max(0.0, 1.0 - response)) if math.isfinite(response) else None
    return source_xy, refined_reference, diagnostic, "phase_correlation_subpixel"


def refine_correspondence(
    source_xy: tuple[float, float],
    reference_xy: tuple[float, float],
    confidence: float,
    dem_query: Any | None,
    camera_geometry: Any | None,
    threshold: float = 0.75,
    source_image: np.ndarray | None = None,
    reference_image: np.ndarray | None = None,
    physical_projector: Callable[[tuple[float, float], Any], tuple[float, float]] | None = None,
) -> RefinementResult:
    dem_valid = bool(dem_query is not None and getattr(dem_query, "valid", False))
    physical = bool(camera_geometry is not None and getattr(getattr(camera_geometry, "capability", None), "exact_physical_projection_available", False))
    branch = select_refinement_branch(confidence, dem_valid, physical, threshold)
    final_source, final_reference = source_xy, reference_xy
    local_error: float | None = None
    physical_residual: float | None = None
    physical_used = False
    reason = ""
    if branch == "dem_physical" and physical_projector is not None and dem_query is not None:
        try:
            predicted = physical_projector(source_xy, dem_query)
            physical_residual = float(np.linalg.norm(np.asarray(predicted) - np.asarray(reference_xy)))
            final_reference = (float(predicted[0]), float(predicted[1]))
            physical_used = True
            reason = "physical_dem_projection_used"
        except Exception as error:
            branch = "classical_fallback"
            reason = f"physical_projection_failed:{type(error).__name__}"
    elif branch == "dem_physical":
        branch = "classical_fallback"
        reason = "physical_projector_not_supplied"
    if source_image is not None and reference_image is not None:
        final_source, final_reference, local_error, local_reason = local_subpixel_refine(source_image, reference_image, final_source, final_reference)
        reason = reason or local_reason
        method = "physics_dem_then_classical_subpixel" if physical_used else ("classical_subpixel" if branch == "classical" else "classical_fallback_subpixel")
    else:
        method = "physical_dem" if physical_used else ("classical" if branch == "classical" else "classical_fallback")
        reason = reason or "image_patches_not_supplied"
    return RefinementResult(
        source_x=float(source_xy[0]),
        source_y=float(source_xy[1]),
        reference_x=float(reference_xy[0]),
        reference_y=float(reference_xy[1]),
        refined_source_x=float(final_source[0]),
        refined_source_y=float(final_source[1]),
        refined_reference_x=float(final_reference[0]),
        refined_reference_y=float(final_reference[1]),
        refinement_method=method,
        reprojection_error=local_error,
        physical_residual=physical_residual,
        confidence=float(confidence),
        status="available" if physical_used or branch == "classical" or source_image is not None else "retained",
        reason=reason,
        physical_geometry_used=physical_used,
    )
