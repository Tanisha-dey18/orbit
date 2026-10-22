"""Shared FastAPI dependencies and upload helpers."""
from __future__ import annotations

import json
from functools import lru_cache

from fastapi import UploadFile
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.errors import ImageValidationError, OrbitError, PayloadTooLargeError
from app.schemas.analysis import AnalysisParams
from app.services.storage import ArtifactStore


@lru_cache
def _store_for(path: str) -> ArtifactStore:
    return ArtifactStore(path)


def get_store() -> ArtifactStore:
    return _store_for(str(get_settings().artifact_dir))


async def read_upload(file: UploadFile | None, label: str) -> tuple[bytes, str | None]:
    """Read an upload with a hard byte limit (reads at most limit+1 bytes)."""
    if file is None:
        raise ImageValidationError(f"Missing {label} image.")
    limit = get_settings().max_upload_bytes
    data = await file.read(limit + 1)
    await file.close()  # Starlette spools large uploads to a temp file; closing removes it
    if len(data) > limit:
        raise PayloadTooLargeError(f"The {label} image exceeds the {get_settings().max_upload_mb:g} MB upload limit.")
    if not data:
        raise ImageValidationError(f"The {label} image is empty.")
    return data, file.filename


def parse_params(raw: str | None) -> AnalysisParams:
    try:
        payload = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            raise ValueError("params must be a JSON object")
        return AnalysisParams(**payload)
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        msg = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
        raise OrbitError(f"Invalid analysis parameters: {msg}", code="invalid_params", status_code=422)
