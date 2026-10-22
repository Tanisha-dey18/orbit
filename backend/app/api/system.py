"""Health, model status and bundled sample endpoints."""
from __future__ import annotations

import json
import re

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.errors import NotFoundError
from app.services.classical import CLASSICAL_METHODS
from app.services.deep import model_service
from app.services.geo import rasterio_available
from app.services.interpretation import CONTEXTS

router = APIRouter(prefix="/api", tags=["system"])
VERSION = "1.0.0"


@router.get("/health")
def health():
    """Backend availability is separate from model readiness: the API is healthy even with no model."""
    return {
        "status": "ok",
        "version": VERSION,
        "backend": "available",
        "model": {"status": model_service.status, "ready": model_service.ready},
        "capabilities": {"classical": True, "deep_learning": model_service.ready,
                         "geotiff_georeferencing": rasterio_available()},
    }


@router.get("/models")
def models():
    s = get_settings()
    return {
        "methods": [
            {"id": "classical", "label": "Classical baseline", "available": True, "is_trained_model": False,
             "variants": [{"id": k, "label": v} for k, v in CLASSICAL_METHODS.items()],
             "default_threshold": 0.12,
             "description": "Deterministic image differencing + thresholding + morphology. Not a trained model."},
            {"id": "deep", "label": "Deep learning (FC-Siam-diff)", "available": model_service.ready,
             "is_trained_model": True, "default_threshold": 0.5, "model": model_service.info(),
             "description": "Siamese U-Net producing per-pixel change probabilities. Requires a trained checkpoint."},
        ],
        "contexts": list(CONTEXTS),
        "limits": {"max_upload_mb": s.max_upload_mb, "max_processing_side": s.max_processing_side,
                   "max_batch_pairs": s.max_batch_pairs},
    }


def _manifest() -> list[dict]:
    path = get_settings().samples_dir / "manifest.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("samples", [])


@router.get("/samples")
def list_samples():
    return {"samples": [{k: v for k, v in s.items() if k != "files"} | {"has_ground_truth": "ground_truth" in s["files"]}
                        for s in _manifest()]}


@router.get("/samples/{sample_id}/{kind}")
def sample_file(sample_id: str, kind: str):
    if not re.fullmatch(r"[a-z0-9_]{1,64}", sample_id) or kind not in ("before", "after", "ground_truth"):
        raise NotFoundError("Sample not found.")
    sample = next((s for s in _manifest() if s["id"] == sample_id), None)
    if sample is None or kind not in sample["files"]:
        raise NotFoundError("Sample not found.")
    root = get_settings().samples_dir.resolve()
    path = (root / sample["files"][kind]).resolve()
    if root not in path.parents or not path.is_file():
        raise NotFoundError("Sample not found.")
    return FileResponse(path, media_type="image/png", filename=f"{sample_id}_{kind}.png")
