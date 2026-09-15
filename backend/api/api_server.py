"""FastAPI transport layer for the existing lunar registration pipeline.

The API owns uploads, run isolation, subprocess execution, and safe artifact
serving. Scientific processing remains in ``run_pipeline.py``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.database.mongo_store import config_summary as mongo_config_summary
from backend.database.mongo_store import get as mongo_get
from backend.database.mongo_store import insert as mongo_insert
from backend.database.mongo_store import list_recent as mongo_list_recent
from backend.database.mongo_store import ping as mongo_ping


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs"
PIPELINE_SCRIPT = ROOT / "backend" / "run_pipeline.py"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
PIPELINE_LOCK = asyncio.Lock()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".img", ".jp2"}
ALLOWED_ARTIFACTS = {
    "source_reference.png",
    "candidate_matches.png",
    "ransac_inliers.png",
    "final_tie_points.png",
    "full_image_tie_points.png",
    "matches.png",
    "inliers.png",
    "final_matching_result.png",
    "registered_source.png",
    "overlay.png",
    "difference.png",
    "tie_points.csv",
    "uniform_grid_points.csv",
    "metrics.json",
    "report.md",
    "physical_validation.csv",
}

ARTIFACT_FALLBACKS = {
    "source_reference.png": ("source_reference.png",),
    "candidate_matches.png": ("candidate_matches.png", "matches.png"),
    "ransac_inliers.png": ("ransac_inliers.png", "inliers.png"),
    "final_tie_points.png": ("final_tie_points.png", "inliers.png"),
    "difference.png": ("difference.png",),
}

STAGE_LABELS = {
    "metadata_context_check": "metadata/context check",
    "catalog_localization": "catalog localization",
    "dem_retrieval_validation": "DEM retrieval/validation",
    "multi_scale_illumination_robust_preprocessing": "multi-scale illumination-robust preprocessing",
    "working_scale_selection": "working-scale selection",
    "loftr_or_baseline_matching": "LoFTR/baseline matching",
    "candidate_filtering": "candidate filtering",
    "native_coordinate_conversion": "native-coordinate conversion",
    "uniform_grid_selection": "uniform-grid selection",
    "image_ransac": "RANSAC",
    "confidence_evaluation": "confidence evaluation",
    "adaptive_subpixel_refinement": "adaptive sub-pixel refinement",
    "adaptive_subpixel_refinement_on_ransac_inliers": "adaptive sub-pixel refinement on RANSAC inliers",
    "dem_physical_validation_or_honest_fallback": "DEM/physical validation",
    "native_coordinate_tie_points": "native-coordinate tie points",
    "original_resolution_registration": "original-resolution registration",
    "registration_outputs_and_metrics": "registration outputs and metrics",
}

app = FastAPI(title="Lunar Image Registration API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _safe_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix in ALLOWED_EXTENSIONS:
        return suffix
    content_type = (upload.content_type or "").lower()
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/tiff": ".tif",
        "image/jp2": ".jp2",
    }.get(content_type, ".bin")


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _coerce_csv_value(key: str, value: str) -> Any:
    if value == "":
        return None
    if key == "inlier" or key == "physical_geometry_used":
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
    if key in {
        "source_x", "source_y", "reference_x", "reference_y", "refined_source_x", "refined_source_y",
        "refined_reference_x", "refined_reference_y", "confidence", "match_distance", "reprojection_error",
        "physical_residual", "subpixel_diagnostic",
    }:
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{key: _coerce_csv_value(key, value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _ensure_source_reference(run_dir: Path, output_dir: Path) -> None:
    target = output_dir / "source_reference.png"
    if target.exists():
        return
    source_path = next((run_dir / name for name in ("source.png", "source.jpg", "source.jpeg", "source.tif", "source.tiff", "source.jp2") if (run_dir / name).exists()), None)
    reference_path = next((run_dir / name for name in ("reference.png", "reference.jpg", "reference.jpeg", "reference.tif", "reference.tiff", "reference.jp2") if (run_dir / name).exists()), None)
    if source_path is None or reference_path is None:
        return
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if source is None or reference is None:
        return
    source = cv2.normalize(source, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    reference = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    height = max(source.shape[0], reference.shape[0])
    source_canvas = cv2.copyMakeBorder(source, 0, height - source.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    reference_canvas = cv2.copyMakeBorder(reference, 0, height - reference.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    canvas = cv2.cvtColor(cv2.hconcat([source_canvas, reference_canvas]), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(target), canvas)


def _ensure_difference(run_dir: Path, output_dir: Path) -> None:
    target = output_dir / "difference.png"
    if target.exists():
        return
    reference_path = next((run_dir / name for name in ("reference.png", "reference.jpg", "reference.jpeg", "reference.tif", "reference.tiff", "reference.jp2") if (run_dir / name).exists()), None)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE) if reference_path else None
    registered = cv2.imread(str(output_dir / "registered_source.png"), cv2.IMREAD_GRAYSCALE)
    if reference is None or registered is None:
        return
    if registered.shape != reference.shape:
        registered = cv2.resize(registered, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(reference, registered)
    cv2.imwrite(str(target), cv2.applyColorMap(difference, cv2.COLORMAP_TURBO))


def _resolve_artifact(output_dir: Path, filename: str) -> Path | None:
    candidates = ARTIFACT_FALLBACKS.get(filename, (filename,))
    for candidate in candidates:
        path = output_dir / candidate
        if path.exists() and path.is_file():
            return path
    return None


def _urls(request: Request, run_id: str) -> dict[str, str]:
    base = str(request.base_url).rstrip("/")
    return {
        "source_reference": f"{base}/api/results/{run_id}/source_reference.png",
        "candidate_matches": f"{base}/api/results/{run_id}/candidate_matches.png",
        "ransac_inliers": f"{base}/api/results/{run_id}/ransac_inliers.png",
        "final_tie_points": f"{base}/api/results/{run_id}/final_tie_points.png",
        "full_image_tie_points": f"{base}/api/results/{run_id}/full_image_tie_points.png",
        "registered_source": f"{base}/api/results/{run_id}/registered_source.png",
        "overlay": f"{base}/api/results/{run_id}/overlay.png",
        "difference": f"{base}/api/results/{run_id}/difference.png",
        "tie_points_csv": f"{base}/api/results/{run_id}/tie_points.csv",
        "metrics_json": f"{base}/api/results/{run_id}/metrics.json",
        "report": f"{base}/api/results/{run_id}/report.md",
    }


def _dimension_object(value: Any) -> dict[str, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return {"height": int(value[0]), "width": int(value[1])}
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict) and {"height", "width"}.issubset(value):
        try:
            return {"height": int(value["height"]), "width": int(value["width"])}
        except (TypeError, ValueError):
            return None
    return None


def _image_metadata(
    path: Path,
    original_filename: str | None,
    role: str,
    run_id: str,
    uploaded_at: datetime,
    fallback_dimensions: Any = None,
) -> dict[str, Any]:
    width: int | None = None
    height: int | None = None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None and len(image.shape) >= 2:
        height, width = int(image.shape[0]), int(image.shape[1])
    else:
        dimensions = _dimension_object(fallback_dimensions)
        if dimensions is not None:
            height, width = dimensions["height"], dimensions["width"]

    stored_path = path.relative_to(ROOT).as_posix()
    return {
        "filename": original_filename or path.name,
        "format": path.suffix.lstrip(".").upper() or None,
        "width": width,
        "height": height,
        "file_size": path.stat().st_size if path.exists() else None,
        "role": role,
        "sensor": None,
        "acquisition": None,
        "upload_timestamp": uploaded_at,
        "path": stored_path,
        "image_id": f"{run_id}:{role}",
    }


def _stage_records(metrics: dict[str, Any] | None, api_status: str) -> list[dict[str, Any]]:
    metrics = metrics or {}
    raw_stages = metrics.get("pipeline_stages") or []
    if not isinstance(raw_stages, list):
        return []
    stage_durations = metrics.get("stage_durations")
    if not isinstance(stage_durations, dict):
        stage_durations = {}
    physical_available = metrics.get("physical_geometry_available")
    validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    dem_available = None
    if validation:
        dem_state = validation.get("dem_elevation_validation")
        if dem_state is not None:
            dem_available = dem_state != "unavailable"

    records: list[dict[str, Any]] = []
    for raw_stage in raw_stages:
        stage_key = str(raw_stage)
        stage_name = STAGE_LABELS.get(stage_key, stage_key.replace("_", " "))
        status = "completed" if api_status in {"success", "insufficient_matches"} else "failed"
        if stage_key == "dem_retrieval_validation" and dem_available is False:
            status = "fallback"
        if "physical" in stage_key and physical_available is False:
            status = "fallback"
        duration = stage_durations.get(stage_key)
        if not isinstance(duration, (int, float)):
            duration = None
        records.append({"stage": stage_name, "status": status, "duration_seconds": duration})
    return records


def _output_references(request: Request, run_id: str, output_dir: Path) -> dict[str, str | None]:
    urls = _urls(request, run_id)
    artifact_names = {
        "correspondence_image": "candidate_matches.png",
        "registered_source": "registered_source.png",
        "overlay": "overlay.png",
        "difference": "difference.png",
        "tie_points_csv": "tie_points.csv",
        "metrics_json": "metrics.json",
        "report": "report.md",
    }
    return {
        key: urls[url_key] if _resolve_artifact(output_dir, filename) is not None else None
        for key, filename in artifact_names.items()
        for url_key in ["candidate_matches" if key == "correspondence_image" else key]
    }


def _mongo_document(
    *,
    run_id: str,
    source_path: Path,
    reference_path: Path,
    source_filename: str | None,
    reference_filename: str | None,
    created_at: datetime,
    completed_at: datetime,
    api_status: str,
    metrics: dict[str, Any] | None,
    tie_points: list[dict[str, Any]],
    output_dir: Path,
    request: Request,
    error: str | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    original_source_dimensions = metrics.get("original_source_dimensions")
    original_reference_dimensions = metrics.get("original_reference_dimensions")
    source_image = _image_metadata(source_path, source_filename, "source", run_id, created_at, original_source_dimensions)
    reference_image = _image_metadata(reference_path, reference_filename, "reference", run_id, created_at, original_reference_dimensions)
    validation = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    dem_available = None
    if validation:
        dem_state = validation.get("dem_elevation_validation")
        if dem_state is not None:
            dem_available = dem_state != "unavailable"
    physical_available = metrics.get("physical_geometry_available")
    physical_refinement_used = any(row.get("physical_geometry_used") is True for row in tie_points)
    final_tie_points = []
    for index, row in enumerate(tie_points, start=1):
        final_tie_points.append({
            "point_id": index,
            "source_x": row.get("source_x"),
            "source_y": row.get("source_y"),
            "reference_x": row.get("reference_x"),
            "reference_y": row.get("reference_y"),
            "confidence": row.get("confidence"),
            "match_distance": row.get("match_distance"),
            "reprojection_error": row.get("reprojection_error"),
            "inlier": row.get("inlier"),
        })

    stored_status = "completed" if api_status == "success" else api_status
    if api_status == "failed":
        stored_status = "failed"
    return {
        "run_id": run_id,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": stored_status,
        "api_status": api_status,
        "method": metrics.get("method"),
        "source_image_id": source_image["image_id"],
        "reference_image_id": reference_image["image_id"],
        "source_image": source_image,
        "reference_image": reference_image,
        "processing": {
            "method": metrics.get("method"),
            "original_source_dimensions": _dimension_object(original_source_dimensions),
            "original_reference_dimensions": _dimension_object(original_reference_dimensions),
            "working_source_dimensions": _dimension_object(metrics.get("working_source_dimensions")),
            "working_reference_dimensions": _dimension_object(metrics.get("working_reference_dimensions")),
            "scale_factor": metrics.get("working_scale_factor"),
            "device": metrics.get("device"),
            "preprocessing_method": metrics.get("preprocessing_method"),
            "preprocessing_stages": [
                stage for stage in metrics.get("pipeline_stages", [])
                if "preprocess" in str(stage) or "scale" in str(stage)
            ],
            "multi_scale": {
                "selected_target_max_dimension": metrics.get("working_scale_selected"),
                "attempts": metrics.get("working_scale_attempts"),
            },
            "dem_available": dem_available,
            "physical_geometry_available": physical_available,
            "physical_refinement_used": physical_refinement_used,
        },
        "matching": {
            "source_keypoints": metrics.get("source_keypoints"),
            "reference_keypoints": metrics.get("reference_keypoints"),
            "raw_loftr_candidate_matches": metrics.get("raw_working_matches"),
            "filtered_candidate_matches": metrics.get("native_candidate_matches_before_grid"),
            "uniform_grid_selected_matches": metrics.get("grid_selected_matches"),
            "final_tie_points": len(final_tie_points),
            "ransac_inliers": metrics.get("ransac_inliers"),
            "ransac_outliers": metrics.get("ransac_outliers"),
            "inlier_ratio": metrics.get("inlier_ratio"),
            "mean_confidence": metrics.get("mean_match_confidence"),
            "minimum_confidence": metrics.get("minimum_match_confidence"),
        },
        "validation": {
            "mean_reprojection_error": metrics.get("mean_reprojection_error_px"),
            "median_reprojection_error": metrics.get("median_reprojection_error_px"),
            "maximum_reprojection_error": metrics.get("maximum_reprojection_error_px"),
            "reprojection_rmse": metrics.get("rmse_reprojection_error_px"),
            "ground_truth_rmse": metrics.get("ground_truth_rmse_px"),
            "physical_residual": metrics.get("physical_reprojection_error"),
            "registration_status": metrics.get("registration_quality"),
            "validation_status": validation or None,
        },
        "tie_points": final_tie_points,
        "outputs": _output_references(request, run_id, output_dir),
        "runtime": {
            "loftr_seconds": metrics.get("loftr_runtime_seconds"),
            "preprocessing_seconds": metrics.get("preprocessing_runtime_seconds"),
            "refinement_seconds": metrics.get("refinement_runtime_seconds"),
            "registration_seconds": metrics.get("registration_runtime_seconds"),
            "total_pipeline_seconds": metrics.get("total_pipeline_runtime_seconds"),
            "api_processing_seconds": None,
            "device": metrics.get("device"),
        },
        "pipeline_stages": _stage_records(metrics, api_status),
        "error": error,
    }


def _persist_document(document: dict[str, Any]) -> dict[str, Any]:
    try:
        return mongo_insert(document)
    except Exception as error:
        return {
            "status": "unavailable",
            "database": mongo_config_summary()["database"],
            "collection": mongo_config_summary()["collection"],
            "error": f"{type(error).__name__}: {error}",
        }


def _pipeline_command(source: Path, reference: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--source",
        str(source),
        "--reference",
        str(reference),
        "--output-dir",
        str(output_dir),
        "--method",
        "loftr",
        "--max-dimension",
        "768",
        "--pipeline-mode",
        "legacy",
    ]


def _run_pipeline(source: Path, reference: Path, output_dir: Path, run_dir: Path) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        _pipeline_command(source, reference, output_dir),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    elapsed = time.perf_counter() - started
    (run_dir / "pipeline_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "pipeline_stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, completed.stdout, completed.stderr, elapsed


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        mongo_status: dict[str, Any] = mongo_ping()
    except Exception as error:
        mongo_status = {
            "status": "unavailable",
            "database": mongo_config_summary()["database"],
            "collection": mongo_config_summary()["collection"],
            "error": f"{type(error).__name__}: {error}",
        }
    return {"status": "ok", "pipeline": str(PIPELINE_SCRIPT.name), "mongodb": mongo_status}


@app.post("/api/register")
async def register(request: Request, source: UploadFile = File(...), reference: UploadFile = File(...)) -> JSONResponse:
    if PIPELINE_LOCK.locked():
        return JSONResponse(
            status_code=409,
            content={
                "status": "busy",
                "error": "A registration is already running. Please wait for it to finish before starting another one.",
            },
        )
    run_id = uuid4().hex
    created_at = datetime.now(timezone.utc)
    source_filename = source.filename
    reference_filename = reference.filename
    run_dir = RUNS_ROOT / run_id
    output_dir = run_dir / "outputs"
    run_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = run_dir / f"source{_safe_suffix(source)}"
    reference_path = run_dir / f"reference{_safe_suffix(reference)}"
    await asyncio.gather(_save_upload(source, source_path), _save_upload(reference, reference_path))

    async with PIPELINE_LOCK:
        try:
            return_code, stdout, stderr, elapsed = await asyncio.to_thread(_run_pipeline, source_path, reference_path, output_dir, run_dir)
        except subprocess.TimeoutExpired as error:
            elapsed = 1800.0
            (run_dir / "pipeline_stderr.log").write_text(f"Pipeline timed out after {elapsed:.1f} seconds.\n{error}", encoding="utf-8")
            completed_at = datetime.now(timezone.utc)
            document = _mongo_document(
                run_id=run_id,
                source_path=source_path,
                reference_path=reference_path,
                source_filename=source_filename,
                reference_filename=reference_filename,
                created_at=created_at,
                completed_at=completed_at,
                api_status="failed",
                metrics=None,
                tie_points=[],
                output_dir=output_dir,
                request=request,
                error="The registration pipeline timed out.",
            )
            database = _persist_document(document)
            return JSONResponse(status_code=504, content={"run_id": run_id, "status": "failed", "error": "The registration pipeline timed out.", "processing_time_seconds": elapsed, "database": database})
        except Exception as error:  # transport-level failure; preserve a clear API error
            message = f"Backend execution failed: {type(error).__name__}: {error}"
            completed_at = datetime.now(timezone.utc)
            document = _mongo_document(
                run_id=run_id,
                source_path=source_path,
                reference_path=reference_path,
                source_filename=source_filename,
                reference_filename=reference_filename,
                created_at=created_at,
                completed_at=completed_at,
                api_status="failed",
                metrics=None,
                tie_points=[],
                output_dir=output_dir,
                request=request,
                error=message,
            )
            database = _persist_document(document)
            return JSONResponse(status_code=500, content={"run_id": run_id, "status": "failed", "error": message, "database": database})

    metrics = _read_json(output_dir / "metrics.json")
    tie_points = _read_csv(output_dir / "tie_points.csv")
    _ensure_source_reference(run_dir, output_dir)
    _ensure_difference(run_dir, output_dir)
    urls = _urls(request, run_id)

    if return_code != 0:
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"run_pipeline.py exited with code {return_code}"
        completed_at = datetime.now(timezone.utc)
        document = _mongo_document(
            run_id=run_id,
            source_path=source_path,
            reference_path=reference_path,
            source_filename=source_filename,
            reference_filename=reference_filename,
            created_at=created_at,
            completed_at=completed_at,
            api_status="failed",
            metrics=metrics,
            tie_points=tie_points,
            output_dir=output_dir,
            request=request,
            error=detail,
        )
        database = _persist_document(document)
        return JSONResponse(
            status_code=500,
            content={
                "run_id": run_id,
                "status": "failed",
                "error": detail,
                "processing_time_seconds": elapsed,
                "metrics": metrics,
                "tie_points": tie_points,
                "outputs": urls,
                "database": database,
            },
        )

    quality = (metrics or {}).get("registration_quality")
    inliers = int((metrics or {}).get("ransac_inliers") or 0)
    status = "insufficient_matches" if quality == "insufficient_matches" or inliers == 0 else "success"
    completed_at = datetime.now(timezone.utc)
    document = _mongo_document(
        run_id=run_id,
        source_path=source_path,
        reference_path=reference_path,
        source_filename=source_filename,
        reference_filename=reference_filename,
        created_at=created_at,
        completed_at=completed_at,
        api_status=status,
        metrics=metrics,
        tie_points=tie_points,
        output_dir=output_dir,
        request=request,
    )
    database = _persist_document(document)
    return JSONResponse(
        content={
            "run_id": run_id,
            "status": status,
            "processing_time_seconds": elapsed,
            "metrics": metrics,
            "tie_points": tie_points,
            "outputs": urls,
            "database": database,
        }
    )


@app.get("/api/registrations/{run_id}")
def registration_by_id(run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise HTTPException(status_code=404, detail="Registration not found")
    try:
        document = mongo_get(run_id)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"MongoDB unavailable: {type(error).__name__}: {error}") from error
    if document is None:
        raise HTTPException(status_code=404, detail="Registration not found")
    return document


@app.get("/api/registrations")
def registrations(limit: int = 20, skip: int = 0) -> dict[str, Any]:
    if limit < 1 or limit > 100 or skip < 0:
        raise HTTPException(status_code=400, detail="limit must be 1-100 and skip must be non-negative")
    try:
        items = mongo_list_recent(limit=limit, skip=skip)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"MongoDB unavailable: {type(error).__name__}: {error}") from error
    return {"items": items, "limit": limit, "skip": skip, "count": len(items)}


@app.get("/api/results/{run_id}/{filename}")
def result_file(run_id: str, filename: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id) or filename not in ALLOWED_ARTIFACTS:
        raise HTTPException(status_code=404, detail="Result artifact not found")
    output_dir = RUNS_ROOT / run_id / "outputs"
    path = _resolve_artifact(output_dir, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Result artifact not available")
    return FileResponse(path)
