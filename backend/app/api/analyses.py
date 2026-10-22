"""Analysis, artifact, report and evaluation endpoints."""
from __future__ import annotations

import io
import json

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool  # CPU-bound work off the event loop

from app.api.deps import get_store, parse_params, read_upload
from app.core.config import get_settings
from app.core.errors import ImageValidationError, OrbitError
from app.database.models import Analysis
from app.database.session import get_db
from app.schemas.analysis import (AnalysisDetail, AnalysisPage, AnalysisSummary, ArtifactInfo, BatchItemResult,
                                  BatchResponse, DeleteResponse, EvaluationResponse, RethresholdRequest)
from app.services import metrics, pipeline
from app.services.image_io import sniff_format
from app.services.report import build_report
from app.services.storage import ArtifactStore, validate_analysis_id

router = APIRouter(prefix="/api", tags=["analyses"])
PUBLIC_ARTIFACTS = ("before", "after", "intensity", "heatmap", "mask", "overlay", "valid", "results", "report")


def _detail(rec: Analysis) -> AnalysisDetail:
    arts = [ArtifactInfo(artifact_type=a.artifact_type, media_type=a.media_type, size_bytes=a.size_bytes,
                         created_at=a.created_at, url=f"/api/analyses/{rec.id}/artifacts/{a.artifact_type}")
            for a in sorted(rec.artifacts, key=lambda a: a.artifact_type)]
    return AnalysisDetail(summary=AnalysisSummary.model_validate(rec), result=rec.result_json, artifacts=arts)


def _load(db: Session, analysis_id: str) -> Analysis:
    return pipeline.get_analysis(db, validate_analysis_id(analysis_id))


@router.post("/analyze", response_model=AnalysisDetail, status_code=201)
async def analyze(
    before: UploadFile = File(..., description="Earlier image (PNG/JPEG/TIFF)"),
    after: UploadFile = File(..., description="Later image (PNG/JPEG/TIFF)"),
    params: str = Form("{}", description="JSON-encoded AnalysisParams"),
    db: Session = Depends(get_db),
    store: ArtifactStore = Depends(get_store),
):
    p = parse_params(params)
    b_data, b_name = await read_upload(before, "BEFORE")
    a_data, a_name = await read_upload(after, "AFTER")
    rec = await run_in_threadpool(pipeline.run_analysis, db, store, b_data, b_name, a_data, a_name, p)
    return _detail(rec)


@router.post("/analyze/batch", response_model=BatchResponse)
async def analyze_batch(
    befores: list[UploadFile] = File(...),
    afters: list[UploadFile] = File(...),
    params: str = Form("{}"),
    db: Session = Depends(get_db),
    store: ArtifactStore = Depends(get_store),
):
    """Analyse up to ORBIT_MAX_BATCH_PAIRS pairs sequentially; befores[i] is paired with afters[i]."""
    limit = get_settings().max_batch_pairs
    if len(befores) != len(afters):
        raise OrbitError("befores and afters must contain the same number of files.", code="invalid_batch", status_code=422)
    if not 1 <= len(befores) <= limit:
        raise OrbitError(f"Batch must contain between 1 and {limit} pairs.", code="invalid_batch", status_code=422)
    p = parse_params(params)
    results = []
    for i, (bf, af) in enumerate(zip(befores, afters)):
        try:
            b_data, b_name = await read_upload(bf, f"BEFORE #{i + 1}")
            a_data, a_name = await read_upload(af, f"AFTER #{i + 1}")
            rec = await run_in_threadpool(pipeline.run_analysis, db, store, b_data, b_name, a_data, a_name, p)
            results.append(BatchItemResult(index=i, ok=True, analysis_id=rec.id,
                                           changed_pixel_percentage=rec.changed_pixel_percentage))
        except OrbitError as exc:
            results.append(BatchItemResult(index=i, ok=False, error=exc.message))
    return BatchResponse(results=results)


@router.get("/analyses", response_model=AnalysisPage)
def list_analyses(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                  method: str | None = Query(None, pattern="^(classical|deep)$"), db: Session = Depends(get_db)):
    q = select(Analysis)
    cq = select(func.count()).select_from(Analysis)
    if method:
        q, cq = q.where(Analysis.method == method), cq.where(Analysis.method == method)
    total = db.scalar(cq) or 0
    rows = db.scalars(q.order_by(Analysis.created_at.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return AnalysisPage(items=[AnalysisSummary.model_validate(r) for r in rows], total=total, page=page,
                        page_size=page_size)


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    return _detail(_load(db, analysis_id))


@router.delete("/analyses/{analysis_id}", response_model=DeleteResponse)
def delete_analysis(analysis_id: str, db: Session = Depends(get_db), store: ArtifactStore = Depends(get_store)):
    """Deletes the database record, its artifact rows (cascade) AND the analysis's artifact directory."""
    rec = _load(db, analysis_id)
    db.delete(rec)
    db.commit()
    files = store.delete_analysis(analysis_id)
    return DeleteResponse(id=analysis_id, deleted_records=1, deleted_files=files)


@router.post("/analyses/{analysis_id}/rethreshold", response_model=AnalysisDetail)
def rethreshold(analysis_id: str, req: RethresholdRequest, db: Session = Depends(get_db),
                store: ArtifactStore = Depends(get_store)):
    """Recompute mask/statistics at a new threshold from the stored intensity map (no model re-run)."""
    return _detail(pipeline.rethreshold(db, store, _load(db, analysis_id), req))


@router.get("/analyses/{analysis_id}/artifacts", response_model=list[ArtifactInfo])
def list_artifacts(analysis_id: str, db: Session = Depends(get_db)):
    return _detail(_load(db, analysis_id)).artifacts


def _serve(rec: Analysis, store: ArtifactStore, kind: str, download: bool) -> FileResponse:
    art = pipeline.artifact_record(rec, kind)
    path = store.path_for(rec.id, art.stored_filename)
    ext = art.stored_filename.rsplit(".", 1)[-1]
    return FileResponse(path, media_type=art.media_type,
                        filename=f"orbit_{rec.id[:8]}_{kind}.{ext}" if download else None,
                        content_disposition_type="attachment" if download else "inline",
                        headers={"Cache-Control": "private, max-age=60"})


def _report(rec: Analysis, store: ArtifactStore, db: Session):
    try:
        pipeline.artifact_record(rec, "report")
    except OrbitError:
        pdf = build_report(rec, store)
        pipeline.replace_artifact(db, store, rec, "report", data=pdf)
        db.commit()
        db.refresh(rec)


@router.get("/analyses/{analysis_id}/artifacts/{kind}")
def get_artifact(analysis_id: str, kind: str, download: bool = False, db: Session = Depends(get_db),
                 store: ArtifactStore = Depends(get_store)):
    if kind not in PUBLIC_ARTIFACTS:
        raise OrbitError("Unknown artifact type.", code="not_found", status_code=404)
    rec = _load(db, analysis_id)
    if kind == "report":
        _report(rec, store, db)
    return _serve(rec, store, kind, download)


@router.get("/analyses/{analysis_id}/report")
def get_report(analysis_id: str, db: Session = Depends(get_db), store: ArtifactStore = Depends(get_store)):
    """Generate (or reuse the cached) PDF report and download it."""
    rec = _load(db, analysis_id)
    _report(rec, store, db)
    return _serve(rec, store, "report", True)


@router.get("/analyses/{analysis_id}/mask")
def get_mask(analysis_id: str, download: bool = False, db: Session = Depends(get_db),
             store: ArtifactStore = Depends(get_store)):
    return _serve(_load(db, analysis_id), store, "mask", download)


@router.get("/analyses/{analysis_id}/overlay")
def get_overlay(analysis_id: str, download: bool = False, db: Session = Depends(get_db),
                store: ArtifactStore = Depends(get_store)):
    return _serve(_load(db, analysis_id), store, "overlay", download)


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    analysis_id: str = Form(...),
    ground_truth: UploadFile = File(..., description="Binary mask PNG/TIFF: >127 = change"),
    db: Session = Depends(get_db),
    store: ArtifactStore = Depends(get_store),
):
    """Compare a stored change mask with an uploaded ground-truth mask (pixel-level metrics)."""
    rec = _load(db, analysis_id)
    data, name = await read_upload(ground_truth, "ground-truth")
    if sniff_format(data) not in ("PNG", "TIFF"):
        raise ImageValidationError("Ground truth must be a lossless PNG or TIFF mask.")
    try:
        gt = np.array(Image.open(io.BytesIO(data)).convert("L")) > 127
    except Exception:
        raise ImageValidationError("Ground-truth mask could not be decoded.")

    warnings = []
    w, h = rec.result_json["processed_size"]
    before_meta = rec.result_json["inputs"]["before"]
    if gt.shape != (h, w):
        if gt.shape == (before_meta["height"], before_meta["width"]):
            gt = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            warnings.append("Ground truth was resized (nearest neighbour) to the processing resolution.")
        else:
            raise ImageValidationError(
                f"Ground truth is {gt.shape[1]}x{gt.shape[0]} but must match the BEFORE image "
                f"({before_meta['width']}x{before_meta['height']}) or the processed size ({w}x{h}).")
    pred = store.read_png(rec.id, pipeline.artifact_record(rec, "mask").stored_filename) > 0
    valid = store.read_png(rec.id, pipeline.artifact_record(rec, "valid").stored_filename) > 0
    m = metrics.binary_metrics(pred, gt, valid)
    if not gt[valid].any():
        warnings.append("Ground truth contains no change pixels; precision/recall/F1/IoU may be undefined.")
    if rec.result_json["alignment"].get("status") == "aligned":
        warnings.append("The AFTER image was re-registered; ground truth is assumed to be in the BEFORE image frame.")
    resp = EvaluationResponse(analysis_id=rec.id, threshold=rec.threshold, method=rec.method,
                              evaluated_pixels=int(valid.sum()), excluded_pixels=int((~valid).sum()),
                              metrics=m, warnings=warnings)
    res = dict(rec.result_json)
    res["evaluation"] = resp.model_dump() | {"ground_truth_filename": (name or "ground_truth")[-120:]}
    rec.result_json = res
    pipeline.replace_artifact(db, store, rec, "results", data=json.dumps(res, indent=2).encode())
    for a in list(rec.artifacts):
        if a.artifact_type == "report":
            store.delete_file(rec.id, a.stored_filename)
            rec.artifacts.remove(a)
            db.delete(a)
    db.commit()
    return resp
