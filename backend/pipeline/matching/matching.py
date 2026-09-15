"""Reusable SIFT/LoFTR matching, spatial selection, and RANSAC utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.feature import LoFTR


@dataclass
class MatchSet:
    source_points: np.ndarray
    reference_points: np.ndarray
    confidence: np.ndarray
    method: str
    source_keypoints: int
    reference_keypoints: int
    metadata: dict[str, Any]

    @property
    def count(self) -> int:
        return int(len(self.source_points))


@dataclass
class RansacSet:
    homography: np.ndarray | None
    inlier_mask: np.ndarray
    reprojection_errors_px: np.ndarray
    threshold_px: float
    model: str = "homography_ransac"

    @property
    def inliers(self) -> int:
        return int(self.inlier_mask.sum())

    @property
    def outliers(self) -> int:
        return int(len(self.inlier_mask) - self.inliers)


def sift_matches(source: np.ndarray, reference: np.ndarray, ratio: float = 0.75, nfeatures: int = 2500) -> MatchSet:
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints0, descriptors0 = sift.detectAndCompute(source, None)
    keypoints1, descriptors1 = sift.detectAndCompute(reference, None)
    if descriptors0 is None or descriptors1 is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return MatchSet(empty, empty, np.empty((0,), dtype=np.float32), "SIFT_BF_RATIO", len(keypoints0), len(keypoints1), {"ratio": ratio})
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors0, descriptors1, k=2)
    good = [pair[0] for pair in candidates if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    points0 = np.asarray([keypoints0[match.queryIdx].pt for match in good], dtype=np.float32).reshape(-1, 2)
    points1 = np.asarray([keypoints1[match.trainIdx].pt for match in good], dtype=np.float32).reshape(-1, 2)
    distances = np.asarray([match.distance for match in good], dtype=np.float32)
    confidence = 1.0 / (1.0 + distances)
    return MatchSet(points0, points1, confidence, "SIFT_BF_RATIO", len(keypoints0), len(keypoints1), {"ratio": ratio, "mean_descriptor_distance": float(distances.mean()) if len(distances) else None})


def rift2_status() -> dict[str, Any]:
    available = bool(hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "RIFT_create"))
    return {"available": available, "implementation": "OpenCV xfeatures2d.RIFT_create" if available else None, "reason": None if available else "RIFT2 is not available in the installed OpenCV build"}


def load_loftr(checkpoint: str | Path, device: torch.device) -> LoFTR:
    model = LoFTR(pretrained=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _loftr_tensor(image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[:2]
    padded_height = int(math.ceil(height / 8) * 8)
    padded_width = int(math.ceil(width / 8) * 8)
    padded = np.zeros((padded_height, padded_width), dtype=np.float32)
    padded[:height, :width] = image.astype(np.float32) / 255.0
    return torch.from_numpy(padded)[None, None], (height, width)


def loftr_matches(model: LoFTR, source: np.ndarray, reference: np.ndarray, device: torch.device, confidence_threshold: float = 0.0) -> MatchSet:
    tensor0, shape0 = _loftr_tensor(source)
    tensor1, shape1 = _loftr_tensor(reference)
    with torch.inference_mode():
        output = model({"image0": tensor0.to(device), "image1": tensor1.to(device)})
    points0 = output.get("keypoints0", torch.empty((0, 2), device=device)).detach().cpu().numpy().reshape(-1, 2)
    points1 = output.get("keypoints1", torch.empty((0, 2), device=device)).detach().cpu().numpy().reshape(-1, 2)
    confidence = output.get("confidence", torch.empty((0,), device=device)).detach().cpu().numpy().reshape(-1)
    keep = confidence >= confidence_threshold
    keep &= (points0[:, 0] >= 0) & (points0[:, 0] < shape0[1]) & (points0[:, 1] >= 0) & (points0[:, 1] < shape0[0]) if len(points0) else np.zeros((0,), dtype=bool)
    keep &= (points1[:, 0] >= 0) & (points1[:, 0] < shape1[1]) & (points1[:, 1] >= 0) & (points1[:, 1] < shape1[0]) if len(points1) else np.zeros((0,), dtype=bool)
    return MatchSet(points0[keep], points1[keep], confidence[keep], "LoFTR", int(len(points0)), int(len(points1)), {"confidence_threshold": confidence_threshold})


def _nearest_2d(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest reference indices and Euclidean distances for 2-D points."""

    if len(query) == 0 or len(reference) == 0:
        return np.empty((len(query),), dtype=np.int64), np.full((len(query),), np.inf, dtype=np.float32)
    distances = np.sum((query[:, None, :].astype(np.float32) - reference[None, :, :].astype(np.float32)) ** 2, axis=2)
    indices = np.argmin(distances, axis=1)
    return indices.astype(np.int64), np.sqrt(distances[np.arange(len(query)), indices]).astype(np.float32)


def mutual_consistency_filter(forward: MatchSet, reverse: MatchSet, radius_px: float = 3.0) -> MatchSet:
    """Keep only forward matches confirmed by a reverse match.

    ``forward`` maps source→reference and ``reverse`` maps reference→source.
    A pair is retained only when both endpoints agree within ``radius_px`` in
    the working image.  This is a geometric mutual-nearest check, not a
    fabricated correspondence or a homography-based acceptance rule.
    """

    if not forward.count or not reverse.count:
        empty = np.empty((0, 2), dtype=np.float32)
        return MatchSet(
            empty,
            empty,
            np.empty((0,), dtype=np.float32),
            forward.method,
            forward.source_keypoints,
            forward.reference_keypoints,
            {
                **forward.metadata,
                "reverse_matches": reverse.count,
                "mutual_radius_px": radius_px,
                "mutual_matches": 0,
            },
        )
    reverse_indices, reference_distances = _nearest_2d(forward.reference_points, reverse.source_points)
    reverse_source_points = reverse.reference_points[reverse_indices]
    source_distances = np.linalg.norm(forward.source_points - reverse_source_points, axis=1)
    keep = (reference_distances <= float(radius_px)) & (source_distances <= float(radius_px))
    if not np.any(keep):
        empty = np.empty((0, 2), dtype=np.float32)
        return MatchSet(
            empty,
            empty,
            np.empty((0,), dtype=np.float32),
            forward.method,
            forward.source_keypoints,
            forward.reference_keypoints,
            {
                **forward.metadata,
                "reverse_matches": reverse.count,
                "mutual_radius_px": radius_px,
                "mutual_matches": 0,
            },
        )
    reverse_confidence = reverse.confidence[reverse_indices]
    confidence = np.minimum(forward.confidence, reverse_confidence)
    return MatchSet(
        forward.source_points[keep],
        forward.reference_points[keep],
        confidence[keep],
        forward.method,
        forward.source_keypoints,
        forward.reference_keypoints,
        {
            **forward.metadata,
            "reverse_matches": reverse.count,
            "mutual_radius_px": radius_px,
            "mutual_matches": int(keep.sum()),
            "forward_matches": forward.count,
        },
    )


def deduplicate_matches(matches: MatchSet, radius_px: float = 3.0) -> MatchSet:
    """Remove near-identical endpoint pairs, preserving strongest confidence."""

    if matches.count < 2:
        return matches
    order = np.argsort(matches.confidence)[::-1]
    chosen: list[int] = []
    radius_sq = float(radius_px) ** 2
    for index in order:
        source_delta = matches.source_points[chosen] - matches.source_points[index] if chosen else np.empty((0, 2), dtype=np.float32)
        reference_delta = matches.reference_points[chosen] - matches.reference_points[index] if chosen else np.empty((0, 2), dtype=np.float32)
        duplicate = bool(
            chosen
            and np.any(
                (np.sum(source_delta * source_delta, axis=1) <= radius_sq)
                & (np.sum(reference_delta * reference_delta, axis=1) <= radius_sq)
            )
        )
        if not duplicate:
            chosen.append(int(index))
    selected = np.asarray(chosen, dtype=np.int64)
    return MatchSet(
        matches.source_points[selected],
        matches.reference_points[selected],
        matches.confidence[selected],
        matches.method,
        matches.source_keypoints,
        matches.reference_keypoints,
        {**matches.metadata, "dedup_radius_px": radius_px, "deduplicated_matches": int(len(selected))},
    )


def filter_match_candidates(
    matches: MatchSet,
    source_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    confidence_threshold: float = 0.2,
    dedup_radius_px: float = 3.0,
) -> MatchSet:
    """Apply confidence, finite/bounds, and near-duplicate filtering."""

    if not matches.count:
        return matches
    source_height, source_width = source_shape
    reference_height, reference_width = reference_shape
    keep = np.isfinite(matches.source_points).all(axis=1) & np.isfinite(matches.reference_points).all(axis=1)
    keep &= matches.confidence >= float(confidence_threshold)
    keep &= (matches.source_points[:, 0] >= 0) & (matches.source_points[:, 0] < source_width)
    keep &= (matches.source_points[:, 1] >= 0) & (matches.source_points[:, 1] < source_height)
    keep &= (matches.reference_points[:, 0] >= 0) & (matches.reference_points[:, 0] < reference_width)
    keep &= (matches.reference_points[:, 1] >= 0) & (matches.reference_points[:, 1] < reference_height)
    filtered = MatchSet(
        matches.source_points[keep],
        matches.reference_points[keep],
        matches.confidence[keep],
        matches.method,
        matches.source_keypoints,
        matches.reference_keypoints,
        {**matches.metadata, "confidence_threshold": float(confidence_threshold), "confidence_filtered_matches": int(keep.sum())},
    )
    return deduplicate_matches(filtered, dedup_radius_px)


def bidirectional_match(
    method: str,
    source: np.ndarray,
    reference: np.ndarray,
    model: LoFTR | None,
    device: torch.device,
    confidence_threshold: float = 0.2,
    mutual_radius_px: float = 3.0,
    dedup_radius_px: float = 3.0,
) -> MatchSet:
    """Run one working-scale matcher in both directions and enforce mutuality."""

    if method == "sift":
        forward = sift_matches(source, reference)
        reverse = sift_matches(reference, source)
    else:
        forward = loftr_matches(model, source, reference, device, confidence_threshold)
        reverse = loftr_matches(model, reference, source, device, confidence_threshold)
    mutual = mutual_consistency_filter(forward, reverse, mutual_radius_px)
    filtered = filter_match_candidates(mutual, source.shape[:2], reference.shape[:2], confidence_threshold, dedup_radius_px)
    return MatchSet(
        filtered.source_points,
        filtered.reference_points,
        filtered.confidence,
        filtered.method,
        max(forward.source_keypoints, reverse.reference_keypoints),
        max(forward.reference_keypoints, reverse.source_keypoints),
        {
            **filtered.metadata,
            "forward_raw_matches": forward.count,
            "reverse_raw_matches": reverse.count,
            "mutual_matches_before_filtering": mutual.count,
            "final_filtered_matches": filtered.count,
        },
    )


def adaptive_bidirectional_match(
    method: str,
    source: np.ndarray,
    reference: np.ndarray,
    model: LoFTR | None,
    device: torch.device,
    confidence_threshold: float = 0.2,
    mutual_radius_px: float = 3.0,
    dedup_radius_px: float = 3.0,
    fallback_scale: float = 0.5,
    fallback_match_count: int = 8,
) -> MatchSet:
    """Use a lower internal scale only when the primary pass is sparse."""

    primary = bidirectional_match(
        method,
        source,
        reference,
        model,
        device,
        confidence_threshold,
        mutual_radius_px,
        dedup_radius_px,
    )
    results = [primary]
    fallback_used = False
    if primary.count < int(fallback_match_count) and 0.0 < float(fallback_scale) < 1.0:
        fallback_used = True
        source_scaled = cv2.resize(source, None, fx=float(fallback_scale), fy=float(fallback_scale), interpolation=cv2.INTER_AREA)
        reference_scaled = cv2.resize(reference, None, fx=float(fallback_scale), fy=float(fallback_scale), interpolation=cv2.INTER_AREA)
        fallback = bidirectional_match(
            method,
            source_scaled,
            reference_scaled,
            model,
            device,
            confidence_threshold,
            mutual_radius_px,
            dedup_radius_px,
        )
        fallback.source_points = fallback.source_points / float(fallback_scale)
        fallback.reference_points = fallback.reference_points / float(fallback_scale)
        results.append(fallback)
    merged = merge_match_sets(results, dedup_radius_px=dedup_radius_px)
    metadata = {
        **merged.metadata,
        "forward_raw_matches": int(sum(int(item.metadata.get("forward_raw_matches", 0)) for item in results)),
        "reverse_raw_matches": int(sum(int(item.metadata.get("reverse_raw_matches", 0)) for item in results)),
        "mutual_matches_before_filtering": int(sum(int(item.metadata.get("mutual_matches_before_filtering", 0)) for item in results)),
        "internal_scales": [1.0] + ([float(fallback_scale)] if fallback_used else []),
        "internal_fallback_used": fallback_used,
        "primary_matches": primary.count,
        "fallback_trigger_count": int(fallback_match_count),
    }
    return MatchSet(
        merged.source_points,
        merged.reference_points,
        merged.confidence,
        merged.method,
        merged.source_keypoints,
        merged.reference_keypoints,
        metadata,
    )


def full_image_tiled_match(
    method: str,
    source: np.ndarray,
    reference: np.ndarray,
    model: LoFTR | None,
    device: torch.device,
    base_matches: MatchSet,
    initial_homography: np.ndarray | None,
    confidence_threshold: float = 0.2,
    mutual_radius_px: float = 8.0,
    dedup_radius_px: float = 3.0,
    tile_grid: tuple[int, int] = (2, 2),
    tile_overlap: float = 0.25,
) -> MatchSet:
    """Match overlapping tiles covering the complete available image.

    The full-frame pass remains part of the result.  Tiles are only used to
    expose reliable features that a single bounded LoFTR view can miss; no
    coordinates are invented.  When a coarse homography is available, each
    source tile is paired with the automatically projected reference overlap.
    """

    source_height, source_width = source.shape[:2]
    reference_height, reference_width = reference.shape[:2]
    rows, columns = tile_grid

    def starts(length: int, count: int) -> list[tuple[int, int]]:
        edges = np.linspace(0, length, count + 1).round().astype(int)
        result: list[tuple[int, int]] = []
        for index in range(count):
            cell_start, cell_end = int(edges[index]), int(edges[index + 1])
            pad = int(round((cell_end - cell_start) * max(0.0, min(0.45, tile_overlap)) / 2.0))
            result.append((max(0, cell_start - pad), min(length, cell_end + pad)))
        return result

    x_tiles = starts(source_width, columns)
    y_tiles = starts(source_height, rows)
    tile_results = [base_matches]
    attempted = 0
    accepted_tiles = 0
    tile_match_count = 0
    valid_overlap_bbox: list[float] | None = None
    if initial_homography is not None:
        full_corners = np.asarray([[[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]]], dtype=np.float32)
        projected_full = cv2.perspectiveTransform(full_corners, initial_homography).reshape(-1, 2)
        if np.isfinite(projected_full).all():
            valid_overlap_bbox = [
                float(max(0.0, projected_full[:, 0].min())),
                float(max(0.0, projected_full[:, 1].min())),
                float(min(float(reference_width), projected_full[:, 0].max())),
                float(min(float(reference_height), projected_full[:, 1].max())),
            ]

    for y0, y1 in y_tiles:
        for x0, x1 in x_tiles:
            source_corners = np.asarray([[[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]]], dtype=np.float32)
            if initial_homography is not None:
                projected = cv2.perspectiveTransform(source_corners, initial_homography).reshape(-1, 2)
                if not np.isfinite(projected).all():
                    continue
                ref_x0, ref_y0 = projected.min(axis=0)
                ref_x1, ref_y1 = projected.max(axis=0)
                padding = max(12.0, 0.15 * max(float(x1 - x0), float(y1 - y0)))
                ref_x0 -= padding
                ref_y0 -= padding
                ref_x1 += padding
                ref_y1 += padding
            else:
                # A normalized full-frame fallback still covers every image
                # cell when the coarse pass cannot produce a model.
                ref_x0 = x0 / max(1, source_width) * reference_width
                ref_x1 = x1 / max(1, source_width) * reference_width
                ref_y0 = y0 / max(1, source_height) * reference_height
                ref_y1 = y1 / max(1, source_height) * reference_height
            rx0 = max(0, int(np.floor(ref_x0)))
            ry0 = max(0, int(np.floor(ref_y0)))
            rx1 = min(reference_width, int(np.ceil(ref_x1)) + 1)
            ry1 = min(reference_height, int(np.ceil(ref_y1)) + 1)
            attempted += 1
            if x1 - x0 < 32 or y1 - y0 < 32 or rx1 - rx0 < 32 or ry1 - ry0 < 32:
                continue
            tile = bidirectional_match(
                method,
                source[y0:y1, x0:x1],
                reference[ry0:ry1, rx0:rx1],
                model,
                device,
                confidence_threshold,
                mutual_radius_px,
                dedup_radius_px,
            )
            if tile.count:
                tile.source_points = tile.source_points + np.asarray([x0, y0], dtype=np.float32)
                tile.reference_points = tile.reference_points + np.asarray([rx0, ry0], dtype=np.float32)
                tile_results.append(tile)
                accepted_tiles += 1
                tile_match_count += tile.count

    merged = merge_match_sets(tile_results, dedup_radius_px=dedup_radius_px)
    return MatchSet(
        merged.source_points,
        merged.reference_points,
        merged.confidence,
        merged.method,
        merged.source_keypoints,
        merged.reference_keypoints,
        {
            **base_matches.metadata,
            **merged.metadata,
            "full_frame_matches": base_matches.count,
            "tiled_raw_matches": tile_match_count,
            "coverage_tiles_attempted": attempted,
            "coverage_tiles_with_matches": accepted_tiles,
            "coverage_tile_grid": list(tile_grid),
            "coverage_tile_overlap": float(tile_overlap),
            "coverage_full_image": True,
            "valid_overlap_reference_bbox_working": valid_overlap_bbox,
        },
    )


def merge_match_sets(sets: Iterable[MatchSet], dedup_radius_px: float = 2.0) -> MatchSet:
    sets = list(sets)
    if not sets:
        empty = np.empty((0, 2), dtype=np.float32)
        return MatchSet(empty, empty, np.empty((0,), dtype=np.float32), "none", 0, 0, {})
    order = np.argsort(np.concatenate([item.confidence for item in sets]))[::-1]
    source = np.concatenate([item.source_points for item in sets], axis=0)
    reference = np.concatenate([item.reference_points for item in sets], axis=0)
    confidence = np.concatenate([item.confidence for item in sets], axis=0)
    chosen: list[int] = []
    for index in order:
        if all(np.linalg.norm(source[index] - source[other]) > dedup_radius_px or np.linalg.norm(reference[index] - reference[other]) > dedup_radius_px for other in chosen):
            chosen.append(int(index))
    chosen_array = np.asarray(chosen, dtype=np.int64)
    return MatchSet(source[chosen_array], reference[chosen_array], confidence[chosen_array], sets[0].method, max(item.source_keypoints for item in sets), max(item.reference_keypoints for item in sets), {"merged_scales": len(sets)})


def multiscale_match(method: str, source: np.ndarray, reference: np.ndarray, model: LoFTR | None, device: torch.device, scales: tuple[float, ...] = (1.0, 0.5), confidence_threshold: float = 0.0) -> MatchSet:
    results: list[MatchSet] = []
    for scale in scales:
        if scale != 1.0:
            source_scaled = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            reference_scaled = cv2.resize(reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            source_scaled, reference_scaled = source, reference
        result = sift_matches(source_scaled, reference_scaled) if method == "sift" else loftr_matches(model, source_scaled, reference_scaled, device, confidence_threshold)
        result.source_points = result.source_points / scale
        result.reference_points = result.reference_points / scale
        result.metadata["scale"] = scale
        results.append(result)
    return merge_match_sets(results)


def uniform_grid_select(matches: MatchSet, reference_shape: tuple[int, int], grid: tuple[int, int] = (8, 8), max_per_cell: int = 4, min_per_cell: int = 0) -> tuple[MatchSet, dict[str, Any]]:
    rows, columns = grid
    height, width = reference_shape
    cells: dict[tuple[int, int], list[int]] = {(row, column): [] for row in range(rows) for column in range(columns)}
    for index, point in enumerate(matches.reference_points):
        column = min(columns - 1, max(0, int(point[0] / max(1, width) * columns)))
        row = min(rows - 1, max(0, int(point[1] / max(1, height) * rows)))
        cells[(row, column)].append(index)
    selected: list[int] = []
    cell_counts: dict[str, int] = {}
    for key, indices in cells.items():
        indices.sort(key=lambda index: float(matches.confidence[index]), reverse=True)
        chosen = indices if max_per_cell <= 0 else indices[:max_per_cell]
        selected.extend(chosen)
        cell_counts[f"{key[0]},{key[1]}"] = len(chosen)
    selected.sort(key=lambda index: float(matches.confidence[index]), reverse=True)
    selected_array = np.asarray(selected, dtype=np.int64)
    chosen_matches = MatchSet(matches.source_points[selected_array], matches.reference_points[selected_array], matches.confidence[selected_array], matches.method, matches.source_keypoints, matches.reference_keypoints, {**matches.metadata, "grid": list(grid), "max_per_cell": max_per_cell, "min_per_cell": min_per_cell}) if len(selected) else MatchSet(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32), matches.method, matches.source_keypoints, matches.reference_keypoints, matches.metadata)
    return chosen_matches, {"grid": list(grid), "cell_counts": cell_counts, "occupied_cells": sum(value > 0 for value in cell_counts.values()), "total_cells": rows * columns}


def ransac_filter(matches: MatchSet, threshold_px: float = 3.0) -> RansacSet:
    if matches.count < 4:
        return RansacSet(None, np.zeros(matches.count, dtype=bool), np.full(matches.count, np.nan), threshold_px)
    homography, mask = cv2.findHomography(matches.source_points, matches.reference_points, cv2.RANSAC, threshold_px)
    if homography is None or mask is None:
        return RansacSet(None, np.zeros(matches.count, dtype=bool), np.full(matches.count, np.nan), threshold_px)
    projected = cv2.perspectiveTransform(matches.source_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - matches.reference_points, axis=1)
    return RansacSet(homography, mask.ravel().astype(bool), errors, threshold_px)


def adaptive_ransac_threshold(source_scale: tuple[float, float], reference_scale: tuple[float, float], working_noise_px: float = 2.5, minimum_native_px: float = 2.0) -> tuple[float, dict[str, float]]:
    """Derive a native-pixel threshold from working-pixel localization noise."""

    scales = [float(value) for value in (*source_scale, *reference_scale) if float(value) > 0.0]
    native_pixels_per_working_pixel = max(1.0 / value for value in scales) if scales else 1.0
    threshold = max(float(minimum_native_px), float(working_noise_px) * native_pixels_per_working_pixel)
    return threshold, {
        "working_noise_px": float(working_noise_px),
        "native_pixels_per_working_pixel": float(native_pixels_per_working_pixel),
        "threshold_px": float(threshold),
    }


def _ransac_candidate_score(ransac: RansacSet) -> tuple[int, float, float]:
    errors = ransac.reprojection_errors_px[ransac.inlier_mask]
    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("inf")
    ratio = ransac.inliers / max(1, len(ransac.inlier_mask))
    return ransac.inliers, ratio, -rmse


def iterative_ransac_filter(matches: MatchSet, threshold_px: float = 3.0, eligible_mask: np.ndarray | None = None) -> RansacSet:
    """Estimate and refit a homography while retaining only robust inliers.

    The optional eligibility mask is used after sub-pixel refinement so that
    only points accepted by the first RANSAC are refined and reconsidered.
    No threshold relaxation is performed between iterations.
    """

    if matches.count < 4:
        return RansacSet(None, np.zeros(matches.count, dtype=bool), np.full(matches.count, np.nan), threshold_px)
    eligible = np.ones(matches.count, dtype=bool) if eligible_mask is None else np.asarray(eligible_mask, dtype=bool).copy()
    if eligible.shape != (matches.count,) or int(eligible.sum()) < 4:
        return RansacSet(None, np.zeros(matches.count, dtype=bool), np.full(matches.count, np.nan), threshold_px)
    candidate_points = matches.source_points[eligible]
    candidate_references = matches.reference_points[eligible]
    cv2.setRNGSeed(0)
    homography, _ = cv2.findHomography(candidate_points, candidate_references, cv2.RANSAC, float(threshold_px), maxIters=5000, confidence=0.999)
    if homography is None:
        return RansacSet(None, np.zeros(matches.count, dtype=bool), np.full(matches.count, np.nan), threshold_px)

    # Start from the scale-derived bound, then tighten it when the residual
    # distribution demonstrates a lower noise level.  This can never loosen
    # the supplied bound merely to increase the inlier count.
    initial_projected = cv2.perspectiveTransform(candidate_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    initial_errors = np.linalg.norm(initial_projected - candidate_references, axis=1)
    initial_inlier_errors = initial_errors[np.isfinite(initial_errors) & (initial_errors <= float(threshold_px))]
    effective_threshold = float(threshold_px)
    if len(initial_inlier_errors) >= 4:
        median_error = float(np.median(initial_inlier_errors))
        mad = float(np.median(np.abs(initial_inlier_errors - median_error)))
        robust_sigma = 1.4826 * mad
        noise_bound = max(2.0, median_error + 3.0 * robust_sigma)
        effective_threshold = min(effective_threshold, noise_bound)

    best: RansacSet | None = None
    current_homography = homography
    for _ in range(3):
        projected = cv2.perspectiveTransform(matches.source_points.reshape(-1, 1, 2), current_homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - matches.reference_points, axis=1)
        mask = eligible & np.isfinite(errors) & (errors <= effective_threshold)
        candidate = RansacSet(current_homography.copy(), mask, errors, effective_threshold)
        if best is None or _ransac_candidate_score(candidate) > _ransac_candidate_score(best):
            best = candidate
        if int(mask.sum()) < 4:
            break
        refit, _ = cv2.findHomography(matches.source_points[mask], matches.reference_points[mask], cv2.RANSAC, effective_threshold, maxIters=5000, confidence=0.999)
        if refit is None:
            refit, _ = cv2.findHomography(matches.source_points[mask], matches.reference_points[mask], 0)
        if refit is None:
            break
        current_homography = refit
    return best if best is not None else RansacSet(None, np.zeros(matches.count, dtype=bool), np.full(matches.count, np.nan), threshold_px)


def spatial_coverage(points: np.ndarray, shape: tuple[int, int], grid: tuple[int, int] = (8, 8)) -> dict[str, Any]:
    height, width = shape
    if len(points) < 3:
        return {"convex_hull_fraction": 0.0, "occupied_grid_cells": 0, "total_grid_cells": grid[0] * grid[1]}
    hull = cv2.convexHull(points.astype(np.float32))
    area = float(cv2.contourArea(hull))
    cells = set()
    for x, y in points:
        cells.add((min(grid[0] - 1, max(0, int(y / max(1, height) * grid[0]))), min(grid[1] - 1, max(0, int(x / max(1, width) * grid[1])))))
    return {"convex_hull_fraction": area / max(1.0, float(width * height)), "occupied_grid_cells": len(cells), "total_grid_cells": grid[0] * grid[1]}


def registration_metrics(matches: MatchSet, ransac: RansacSet, reference_shape: tuple[int, int]) -> dict[str, Any]:
    inlier_errors = ransac.reprojection_errors_px[ransac.inlier_mask] if len(ransac.inlier_mask) else np.empty((0,))
    return {
        "source_keypoints": matches.source_keypoints,
        "reference_keypoints": matches.reference_keypoints,
        "candidate_matches": matches.count,
        "ransac_inliers": ransac.inliers,
        "ransac_outliers": ransac.outliers,
        "inlier_ratio": ransac.inliers / max(1, matches.count),
        "mean_match_confidence": float(np.mean(matches.confidence)) if matches.count else None,
        "minimum_match_confidence": float(np.min(matches.confidence)) if matches.count else None,
        "mean_reprojection_error_px": float(np.mean(inlier_errors)) if len(inlier_errors) else None,
        "median_reprojection_error_px": float(np.median(inlier_errors)) if len(inlier_errors) else None,
        "maximum_reprojection_error_px": float(np.max(inlier_errors)) if len(inlier_errors) else None,
        "rmse_reprojection_error_px": float(np.sqrt(np.mean(inlier_errors ** 2))) if len(inlier_errors) else None,
        "tie_point_spatial_coverage": spatial_coverage(matches.reference_points[ransac.inlier_mask], reference_shape),
        "geometric_model": ransac.model if ransac.homography is not None else None,
    }
