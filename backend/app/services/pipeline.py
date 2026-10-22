"""End-to-end analysis orchestration.

  load/validate -> (reproject) -> resize -> register -> normalise
  -> change intensity (classical OR deep model) -> threshold -> morphology
  -> statistics -> area -> interpretation -> artifacts -> database
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import ModelUnavailableError, NotFoundError, OrbitError
from app.database.models import Analysis, Artifact
from app.schemas.analysis import AnalysisParams, RethresholdRequest
from app.services import classical, interpretation, metrics, visualization
from app.services.alignment import prepare_pair
from app.services.deep import model_service
from app.services.geo import GeoInfo, estimate_changed_area
from app.services.image_io import LoadedImage, load_image, safe_display_name
from app.services.storage import ARTIFACT_TYPES, ArtifactStore

logger = logging.getLogger("orbit.pipeline")

LIMITATIONS = [
    "changed_pixel_percentage is the share of VALID compared pixels in the analysed image area, "
    "not automatically the share of geographic land area affected.",
    "Classical methods flag any radiometric difference (lighting, season, shadows, haze, sensor) as change.",
    "Deep-learning results are only as reliable as the checkpoint's training data; a model trained on "
    "building change (e.g. LEVIR-CD) is not validated for floods or vegetation.",
    "Change categories are rule-based RGB heuristics, not validated semantic predictions.",
    "Satellite imagery alone may be insufficient to determine the cause or severity of a change.",
]

_slots: threading.BoundedSemaphore | None = None


def _semaphore() -> threading.BoundedSemaphore:
    global _slots
    if _slots is None:
        _slots = threading.BoundedSemaphore(max(1, get_settings().max_concurrent_analyses))
    return _slots


def _geo_to_dict(geo: GeoInfo | None) -> dict | None:
    return None if geo is None else {
        "crs_wkt": geo.crs_wkt, "crs_name": geo.crs_name, "transform": list(geo.transform),
        "width": geo.width, "height": geo.height, "is_projected": geo.is_projected,
        "is_geographic": geo.is_geographic, "linear_unit_m": geo.linear_unit_m,
    }


def _geo_from_dict(d: dict | None) -> GeoInfo | None:
    return None if not d else GeoInfo(**{**d, "transform": tuple(d["transform"])})


def _area_block(mask: np.ndarray, geo: GeoInfo | None, sx: float, sy: float) -> dict:
    area, explanation = estimate_changed_area(mask, geo, sx, sy)
    return {
        "changed_area_m2": round(area, 2) if area is not None else None,
        "changed_area_km2": round(area / 1e6, 6) if area is not None else None,
        "explanation": explanation,
        "crs": geo.crs_name if geo else None,
        "pixel_size_crs_units": list(geo.pixel_size) if geo else None,
    }


def _status(reliable: bool, warnings: list[str]) -> str:
    if not reliable:
        return "unreliable"
    return "completed_with_warnings" if warnings else "completed"


def _quantize(intensity: np.ndarray) -> np.ndarray:
    """Quantise to 1/255 steps so later re-thresholding from the stored PNG is exactly reproducible."""
    return (np.round(np.clip(intensity, 0, 1) * 255.0) / 255.0).astype(np.float32)


def _method_block(params: AnalysisParams) -> dict:
    if params.method == "deep":
        meta = model_service.metadata
        return {
            "type": "deep", "variant": meta.get("arch", "fc_siam_diff"),
            "label": "Deep learning - FC-Siam-diff (Siamese U-Net)", "is_trained_model": True,
            "model": {
                "name": meta.get("model_name"), "version": meta.get("model_version"),
                "dataset": meta.get("dataset"), "epochs_trained": meta.get("epochs_trained"),
                "best_val_recorded_in_checkpoint": meta.get("best_val"), "sha256": meta.get("sha256"),
                "parameters": meta.get("parameters"), "device": model_service.device,
                "created_at": meta.get("created_at"),
            },
            "description": "Pixel-wise change probability from a Siamese U-Net checkpoint.",
        }
    return {
        "type": "classical", "variant": params.classical_method,
        "label": f"Classical baseline - {classical.CLASSICAL_METHODS[params.classical_method]}",
        "is_trained_model": False, "model": None,
        "description": "Deterministic image-processing baseline, NOT a trained AI prediction.",
    }


def _finalize(
    intensity: np.ndarray, valid: np.ndarray, before: np.ndarray, after: np.ndarray, threshold: float,
    morph_kernel: int, min_region_px: int, context: str, reliable: bool, geo: GeoInfo | None,
    sx: float, sy: float,
) -> dict:
    mask = classical.postprocess_mask(intensity, valid, threshold, morph_kernel, min_region_px)
    stats = metrics.change_statistics(mask, valid, intensity)
    return {
        "mask": mask,
        "statistics": stats,
        "area": _area_block(mask, geo, sx, sy),
        "interpretation": interpretation.interpret(before, after, mask, stats, threshold, context, reliable),
    }


def _save(db: Session, store: ArtifactStore, analysis_id: str, kind: str, array=None, data: bytes | None = None):
    if array is not None:
        name, size = store.save_png(analysis_id, kind, array)
    else:
        name, size = store.save_bytes(analysis_id, kind, data or b"")
    db.add(Artifact(analysis_id=analysis_id, artifact_type=kind, stored_filename=name,
                    media_type=ARTIFACT_TYPES[kind], size_bytes=size))


def run_analysis(db: Session, store: ArtifactStore, before_bytes: bytes, before_name: str | None,
                 after_bytes: bytes, after_name: str | None, params: AnalysisParams) -> Analysis:
    s = get_settings()
    t_start = time.perf_counter()
    if params.method == "deep" and not model_service.ready:
        raise ModelUnavailableError(
            "Deep-learning analysis is unavailable: " + model_service.message +
            " Use method='classical' for the baseline.")

    if not _semaphore().acquire(timeout=120):
        raise OrbitError("Server is busy processing other analyses; try again shortly.",
                         code="busy", status_code=503)
    try:
        before = load_image(before_bytes, safe_display_name(before_name, "before"))
        after = load_image(after_bytes, safe_display_name(after_name, "after"))
        return _run(db, store, before, after, params, t_start, s.max_processing_side)
    finally:
        _semaphore().release()


def _run(db: Session, store: ArtifactStore, before: LoadedImage, after: LoadedImage,
         params: AnalysisParams, t_start: float, max_side: int) -> Analysis:
    warnings = [f"BEFORE: {w}" for w in before.warnings] + [f"AFTER: {w}" for w in after.warnings]
    try:
        pair = prepare_pair(before, after, params.alignment, params.normalization, max_side)
    except ValueError as exc:
        raise OrbitError(f"Comparison not possible: {exc}", code="comparison_not_possible", status_code=422)
    warnings += pair.warnings

    # ---- change intensity --------------------------------------------------------------
    t0 = time.perf_counter()
    if params.method == "deep":
        intensity, inference_ms = model_service.predict(pair.before, pair.after)
        if str(model_service.metadata.get("dataset", "")).startswith("synthetic"):
            warnings.append("The loaded checkpoint was trained only on SYNTHETIC smoke-test data; its output "
                            "is not meaningful for real imagery.")
    else:
        intensity = classical.difference_intensity(pair.before, pair.after, params.classical_method, params.blur_sigma)
        inference_ms = (time.perf_counter() - t0) * 1000
    intensity = _quantize(intensity)
    intensity[~pair.valid] = 0.0

    threshold = (classical.otsu_threshold(intensity, pair.valid) if params.threshold_mode == "otsu"
                 else params.effective_threshold())
    fin = _finalize(intensity, pair.valid, pair.before, pair.after, threshold, params.morph_kernel,
                    params.min_region_px, params.context, pair.reliable, pair.geo, pair.scale_x, pair.scale_y)
    stats, mask = fin["statistics"], fin["mask"]
    if stats["changed_pixel_percentage"] > 60:
        warnings.append("More than 60% of pixels changed - this often indicates misalignment, very different "
                        "acquisition conditions or a threshold that is too low.")

    analysis_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc)
    method = _method_block(params)
    h, w = mask.shape
    result = {
        "id": analysis_id,
        "created_at": created.isoformat(),
        "status": _status(pair.reliable, warnings),
        "reliable": pair.reliable,
        "inputs": {"before": before.metadata(), "after": after.metadata()},
        "processed_size": [w, h],
        "scale": [round(pair.scale_x, 6), round(pair.scale_y, 6)],
        "geo": _geo_to_dict(pair.geo),
        "method": method,
        "parameters": params.model_dump(),
        "threshold": {"value": round(threshold, 4), "mode": params.threshold_mode},
        "alignment": pair.alignment,
        "statistics": stats,
        "area": fin["area"],
        "interpretation": fin["interpretation"],
        "histogram": metrics.intensity_histogram(intensity, pair.valid),
        "timing": {"inference_ms": round(inference_ms, 2)},
        "warnings": warnings,
        "limitations": LIMITATIONS,
        "evaluation": None,
    }

    try:
        valid_u8 = (pair.valid.astype(np.uint8) * 255)
        for kind, arr in (
            ("before", pair.before), ("after", pair.after),
            ("intensity", visualization.intensity_image(intensity, pair.valid)),
            ("heatmap", visualization.heatmap(intensity, pair.valid)),
            ("mask", visualization.mask_image(mask)),
            ("overlay", visualization.overlay(pair.after, mask, pair.valid)),
            ("valid", valid_u8),
        ):
            _save(db, store, analysis_id, kind, array=arr)
        total_ms = (time.perf_counter() - t_start) * 1000
        result["timing"]["total_ms"] = round(total_ms, 2)
        _save(db, store, analysis_id, "results", data=json.dumps(result, indent=2).encode())
        rec = Analysis(
            id=analysis_id, created_at=created, before_filename=before.filename, after_filename=after.filename,
            method=params.method, method_variant=method["variant"],
            model_name=(method["model"] or {}).get("name"), model_version=(method["model"] or {}).get("version"),
            status=result["status"], context=params.context, threshold=threshold,
            changed_pixel_count=stats["changed_pixel_count"], valid_pixel_count=stats["valid_pixel_count"],
            changed_pixel_percentage=stats["changed_pixel_percentage"], region_count=stats["region_count"],
            changed_area_m2=fin["area"]["changed_area_m2"], inference_time_ms=round(inference_ms, 2),
            total_time_ms=round(total_ms, 2), warning_summary=" | ".join(warnings)[:4000] or None,
            params_json=params.model_dump(), result_json=result,
        )
        db.add(rec)
        db.commit()
    except Exception:
        db.rollback()
        store.delete_analysis(analysis_id)
        raise
    logger.info("analysis %s done method=%s changed=%.3f%% time=%.0fms", analysis_id, params.method,
                stats["changed_pixel_percentage"], result["timing"]["total_ms"])
    return rec


def get_analysis(db: Session, analysis_id: str) -> Analysis:
    rec = db.get(Analysis, analysis_id)
    if rec is None:
        raise NotFoundError("Analysis not found.")
    return rec


def artifact_record(rec: Analysis, kind: str) -> Artifact:
    for a in rec.artifacts:
        if a.artifact_type == kind:
            return a
    raise NotFoundError(f"Artifact '{kind}' is not available for this analysis.")


def replace_artifact(db: Session, store: ArtifactStore, rec: Analysis, kind: str, array=None, data=None):
    for a in list(rec.artifacts):
        if a.artifact_type == kind:
            store.delete_file(rec.id, a.stored_filename)
            rec.artifacts.remove(a)
            db.delete(a)
    db.flush()
    _save(db, store, rec.id, kind, array=array, data=data)


def rethreshold(db: Session, store: ArtifactStore, rec: Analysis, req: RethresholdRequest) -> Analysis:
    """Recompute mask + statistics from the stored intensity map - no model re-run."""
    res = dict(rec.result_json)
    params = dict(res["parameters"])
    intensity = store.read_png(rec.id, artifact_record(rec, "intensity").stored_filename).astype(np.float32) / 255.0
    valid = store.read_png(rec.id, artifact_record(rec, "valid").stored_filename) > 0
    before = store.read_png(rec.id, artifact_record(rec, "before").stored_filename)
    after = store.read_png(rec.id, artifact_record(rec, "after").stored_filename)

    if req.threshold_mode == "otsu":
        threshold = classical.otsu_threshold(intensity, valid)
    else:
        threshold = req.threshold if req.threshold is not None else rec.threshold
    morph = req.morph_kernel if req.morph_kernel is not None else params["morph_kernel"]
    if morph > 1 and morph % 2 == 0:
        morph += 1
    min_px = req.min_region_px if req.min_region_px is not None else params["min_region_px"]
    sx, sy = res.get("scale", [1.0, 1.0])
    fin = _finalize(intensity, valid, before, after, threshold, morph, min_px, params["context"],
                    res["reliable"], _geo_from_dict(res.get("geo")), sx, sy)
    stats, mask = fin["statistics"], fin["mask"]

    params.update({"threshold": round(threshold, 4), "threshold_mode": req.threshold_mode,
                   "morph_kernel": morph, "min_region_px": min_px})
    res.update({"parameters": params, "threshold": {"value": round(threshold, 4), "mode": req.threshold_mode},
                "statistics": stats, "area": fin["area"], "interpretation": fin["interpretation"],
                "rethresholded_at": datetime.now(timezone.utc).isoformat(), "evaluation": None})

    replace_artifact(db, store, rec, "mask", array=visualization.mask_image(mask))
    replace_artifact(db, store, rec, "overlay", array=visualization.overlay(after, mask, valid))
    replace_artifact(db, store, rec, "results", data=json.dumps(res, indent=2).encode())
    for a in list(rec.artifacts):  # cached report is now stale
        if a.artifact_type == "report":
            store.delete_file(rec.id, a.stored_filename)
            rec.artifacts.remove(a)
            db.delete(a)
    rec.threshold = threshold
    rec.changed_pixel_count = stats["changed_pixel_count"]
    rec.changed_pixel_percentage = stats["changed_pixel_percentage"]
    rec.region_count = stats["region_count"]
    rec.changed_area_m2 = fin["area"]["changed_area_m2"]
    rec.params_json = params
    rec.result_json = res
    db.commit()
    db.refresh(rec)
    return rec
