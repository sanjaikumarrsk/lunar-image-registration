"""Memory-conscious readers for Chandrayaan/Kaguya rasters and previews."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...geometry.pds import parse_pds3_label


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _xml_text(root: ET.Element, name: str) -> str | None:
    for element in root.iter():
        if _local(element.tag) == name.lower() and element.text and element.text.strip():
            return element.text.strip()
    return None


@dataclass(frozen=True)
class RasterSpec:
    path: Path
    label_path: Path | None
    shape: tuple[int, int]
    dtype: np.dtype
    offset_bytes: int = 0
    scale: float = 1.0
    offset: float = 0.0
    dummy: float | None = None
    storage_shape: tuple[int, ...] | None = None
    band_index: int = 0

    @classmethod
    def from_path(cls, path: str | Path, label_path: str | Path | None = None) -> "RasterSpec":
        path = Path(path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Cannot read raster {path}")
            if image.ndim != 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cls(path, None, tuple(int(value) for value in image.shape), image.dtype)
        label = Path(label_path) if label_path is not None else _infer_label(path)
        if label is None or not label.exists():
            raise ValueError(f"A detached PDS label is required for {path}")
        if label.suffix.lower() == ".lbl":
            values = parse_pds3_label(label)
            lines = int(float(values.get("LINE_LAST_PIXEL", values.get("LINES", 0))))
            samples = int(float(values.get("SAMPLE_LAST_PIXEL", values.get("LINE_SAMPLES", 0))))
            sample_type = str(values.get("SAMPLE_TYPE", "")).upper()
            dtype = _dtype_from_sample_type(sample_type, int(float(values.get("SAMPLE_BITS", 16))))
            scale = float(values.get("SCALING_FACTOR", 1.0))
            offset = float(values.get("OFFSET", 0.0))
            dummy = float(values["DUMMY"]) if "DUMMY" in values else None
            return cls(path, label, (lines, samples), dtype, 0, scale, offset, dummy)
        root = ET.parse(label).getroot()
        array = next((element for element in root.iter() if _local(element.tag) in {"array_2d_image", "array_3d_image", "array_3d_spectrum"}), None)
        if array is None:
            raise ValueError(f"No image array in {label}")
        data_type = _xml_text(array, "data_type") or "UnsignedByte"
        bits = int(_xml_text(array, "element_size") or _xml_text(array, "sample_bits") or _bits_from_data_type(data_type))
        dtype = _dtype_from_sample_type(data_type, bits)
        axes: list[tuple[int, int]] = []
        for axis in (element for element in array.iter() if _local(element.tag) == "axis_array"):
            elements = _xml_text(axis, "elements")
            sequence = _xml_text(axis, "sequence_number")
            if elements and sequence:
                axes.append((int(sequence), int(elements)))
        axes.sort()
        if len(axes) < 2:
            raise ValueError(f"Incomplete image axes in {label}")
        offset = int(_xml_text(array, "offset") or 0)
        if len(axes) == 2:
            image_shape = (axes[0][1], axes[1][1])
            storage_shape = image_shape
        else:
            # IIRS spectra are stored as (spectral element, line, sample).
            # A band is selected explicitly by ``band_index``; no averaging
            # or fabricated radiometry is performed.
            image_shape = (axes[1][1], axes[2][1])
            storage_shape = tuple(value for _, value in axes)
        return cls(path, label, image_shape, dtype, offset, 1.0, 0.0, None, storage_shape, 0)

    def array(self) -> np.ndarray:
        if self.path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            image = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Cannot read raster {self.path}")
            return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        storage_shape = self.storage_shape or self.shape
        raw = np.memmap(self.path, dtype=self.dtype, mode="r", offset=self.offset_bytes, shape=storage_shape)
        return raw if len(storage_shape) == 2 else raw[self.band_index]

    def valid_mask(self, array: np.ndarray) -> np.ndarray:
        mask = np.isfinite(array.astype(np.float64, copy=False))
        if self.dummy is not None:
            mask &= array != self.dummy
        return mask


def _bits_from_data_type(data_type: str) -> int:
    match = re.search(r"(\d+)$", data_type)
    return int(match.group(1)) * 8 if match else 8


def _dtype_from_sample_type(sample_type: str, bits: int) -> np.dtype:
    lower = sample_type.lower().replace("_", "")
    if bits == 8:
        return np.dtype("u1")
    signed = lower.startswith("signed") or ("unsigned" not in lower and lower.startswith("int"))
    little = "lsb" in lower or "little" in lower
    kind = "i" if signed else "u"
    endian = "<" if little else ">"
    return np.dtype(endian + kind + str(max(1, bits // 8)))


def _infer_label(path: Path) -> Path | None:
    sidecar = path.with_suffix(".xml")
    if sidecar.exists():
        return sidecar
    if path.parent.name == "tc_ortho":
        candidate = path.parents[1] / "metadata" / f"{path.stem}_img.lbl"
        if candidate.exists():
            return candidate
    return None


def normalize_for_matching(array: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    mask = valid_mask if valid_mask is not None else np.isfinite(values)
    finite = values[mask & np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) * 255.0 / (high - low), 0.0, 255.0).astype(np.uint8)
    normalized[~mask] = 0
    return normalized


def robust_local_contrast(image: np.ndarray) -> np.ndarray:
    """Build an illumination-robust grayscale view for correspondence search.

    The native image is never modified.  CLAHE preserves local crater/ridge
    contrast, a broad background estimate suppresses illumination gradients,
    and a small rotation-invariant gradient contribution protects structural
    edges when sensors have different radiometric responses.
    """

    if image.ndim != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image8 = image if image.dtype == np.uint8 else normalize_for_matching(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    local_contrast = clahe.apply(image8)
    background = cv2.GaussianBlur(local_contrast, (0, 0), 9.0)
    detail = cv2.addWeighted(local_contrast, 1.35, background, -0.35, 128.0)
    detail = np.clip(detail, 0, 255).astype(np.uint8)

    gradient_x = cv2.Scharr(detail, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(detail, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    finite = gradient[np.isfinite(gradient)]
    if finite.size:
        low, high = np.percentile(finite, [5.0, 99.0])
        if high > low:
            gradient = np.clip((gradient - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        else:
            gradient = np.zeros_like(detail)
    else:
        gradient = np.zeros_like(detail)
    # ``preview_raster`` already performs percentile normalization, CLAHE,
    # and broad-background suppression.  Keep that calibrated view dominant;
    # the second pass is intentionally conservative so LoFTR sees the same
    # crater/ridge structures instead of an edge-only image.
    combined = cv2.addWeighted(image8, 0.78, detail, 0.18, 0.0)
    combined = cv2.addWeighted(combined, 0.96, gradient, 0.04, 0.0)
    return np.clip(combined, 0, 255).astype(np.uint8)


def preview_raster(spec: RasterSpec, max_dimension: int = 1024, illumination_robust: bool = True) -> tuple[np.ndarray, float, float]:
    array = spec.array()
    height, width = array.shape[:2]
    target_scale = min(1.0, float(max_dimension) / max(height, width))
    # Do not upscale small inputs.  LoFTR's tensor adapter pads internally to
    # its minimum stride without changing the reported working dimensions.
    target_width = max(1, round(width * target_scale))
    target_height = max(1, round(height * target_scale))

    # PDS products can be hundreds of millions of pixels.  Normalize a
    # decimated view before resizing instead of materializing a full float32
    # copy merely to build a matching preview.  The returned scales are the
    # actual preview-to-input coordinate scales, including rounding.
    if target_scale < 1.0:
        step = max(1, math.ceil(1.0 / target_scale))
        sampled = array[::step, ::step]
        sampled_mask = spec.valid_mask(sampled)
        normalized = normalize_for_matching(sampled, sampled_mask)
        resized = cv2.resize(normalized, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        normalized = normalize_for_matching(array, spec.valid_mask(array))
        resized = normalized if normalized.shape[:2] == (target_height, target_width) else cv2.resize(normalized, (target_width, target_height), interpolation=cv2.INTER_AREA)
    scale_x = target_width / float(width)
    scale_y = target_height / float(height)
    if illumination_robust:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        resized = clahe.apply(resized)
        background = cv2.GaussianBlur(resized, (0, 0), 9.0)
        resized = cv2.normalize(cv2.addWeighted(resized, 1.5, background, -0.5, 128.0), None, 0, 255, cv2.NORM_MINMAX)
    return resized, scale_x, scale_y


def native_to_preview(point_xy: np.ndarray | tuple[float, float], scale_x: float, scale_y: float) -> np.ndarray:
    point = np.asarray(point_xy, dtype=np.float64)
    return point * np.asarray([scale_x, scale_y], dtype=np.float64)


def preview_to_native(point_xy: np.ndarray | tuple[float, float], scale_x: float, scale_y: float) -> np.ndarray:
    point = np.asarray(point_xy, dtype=np.float64)
    return point / np.asarray([scale_x, scale_y], dtype=np.float64)


def write_warped_native_png(spec: RasterSpec, homography_preview: np.ndarray, source_scale: tuple[float, float], reference_shape: tuple[int, int], output_path: str | Path) -> dict[str, Any]:
    """Warp native source pixels; the homography is estimated in preview coordinates."""

    source = spec.array()
    sx, sy = source_scale
    source_to_preview = np.asarray([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    reference_width, reference_height = reference_shape[1], reference_shape[0]
    # A reference preview coordinate is converted back to the native reference
    # grid.  The caller supplies the reference scale through homography already
    # composed by compose_native_homography.
    warped = cv2.warpPerspective(source, homography_preview, (reference_width, reference_height), flags=cv2.INTER_LINEAR)
    if not cv2.imwrite(str(output_path), warped):
        raise IOError(f"Cannot write {output_path}")
    return {"source_native_shape": list(source.shape), "output_shape": [reference_height, reference_width], "dtype": str(source.dtype), "used_native_source": True}


def compose_native_homography(homography_preview: np.ndarray, source_scale: tuple[float, float], reference_scale: tuple[float, float]) -> np.ndarray:
    source_native_to_preview = np.asarray([[source_scale[0], 0.0, 0.0], [0.0, source_scale[1], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    reference_native_to_preview = np.asarray([[reference_scale[0], 0.0, 0.0], [0.0, reference_scale[1], 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.linalg.inv(reference_native_to_preview) @ np.asarray(homography_preview, dtype=np.float64) @ source_native_to_preview
