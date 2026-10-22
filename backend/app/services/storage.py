"""Local artifact storage with path-traversal protection.

Layout:  <ORBIT_ARTIFACT_DIR>/analyses/<analysis_id>/<type>_<random>.<ext>

* Analysis IDs are server-generated 32-char hex UUIDs and validated on every access.
* Stored filenames are random and recorded in the database; clients only ever ask
  for an artifact *type*, never a filename or path.
* Every resolved path is checked to be inside the artifact root.
"""
from __future__ import annotations

import io
import re
import secrets
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.errors import NotFoundError

ANALYSIS_ID_RE = re.compile(r"^[0-9a-f]{32}$")
ARTIFACT_TYPES = {
    "before": "image/png",          # processed (aligned/normalised) BEFORE image
    "after": "image/png",           # processed AFTER image, aligned to BEFORE
    "intensity": "image/png",       # 8-bit change intensity / probability (0-255)
    "heatmap": "image/png",
    "mask": "image/png",
    "overlay": "image/png",
    "valid": "image/png",           # comparison-validity mask
    "report": "application/pdf",
    "results": "application/json",
}
_EXT = {"image/png": "png", "application/pdf": "pdf", "application/json": "json"}


def validate_analysis_id(analysis_id: str) -> str:
    if not ANALYSIS_ID_RE.fullmatch(analysis_id or ""):
        raise NotFoundError("Analysis not found.")
    return analysis_id


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.base = self.root / "analyses"
        self.base.mkdir(parents=True, exist_ok=True)

    def analysis_dir(self, analysis_id: str) -> Path:
        d = (self.base / validate_analysis_id(analysis_id)).resolve()
        if self.base not in d.parents:
            raise NotFoundError("Analysis not found.")
        return d

    def _new_name(self, artifact_type: str) -> str:
        if artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"unknown artifact type {artifact_type}")
        return f"{artifact_type}_{secrets.token_hex(8)}.{_EXT[ARTIFACT_TYPES[artifact_type]]}"

    def save_bytes(self, analysis_id: str, artifact_type: str, data: bytes) -> tuple[str, int]:
        d = self.analysis_dir(analysis_id)
        d.mkdir(parents=True, exist_ok=True)
        name = self._new_name(artifact_type)
        tmp = d / (name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(d / name)  # atomic within the same directory
        return name, len(data)

    def save_png(self, analysis_id: str, artifact_type: str, array: np.ndarray) -> tuple[str, int]:
        buf = io.BytesIO()
        Image.fromarray(array).save(buf, format="PNG", optimize=False, compress_level=6)
        return self.save_bytes(analysis_id, artifact_type, buf.getvalue())

    def path_for(self, analysis_id: str, stored_filename: str) -> Path:
        if not re.fullmatch(r"[a-z_]+_[0-9a-f]{16}\.(png|pdf|json)", stored_filename or ""):
            raise NotFoundError("Artifact not found.")
        p = (self.analysis_dir(analysis_id) / stored_filename).resolve()
        if p.parent != self.analysis_dir(analysis_id) or not p.is_file():
            raise NotFoundError("Artifact not found.")
        return p

    def delete_file(self, analysis_id: str, stored_filename: str) -> None:
        try:
            self.path_for(analysis_id, stored_filename).unlink(missing_ok=True)
        except NotFoundError:
            pass

    def delete_analysis(self, analysis_id: str) -> bool:
        d = self.analysis_dir(analysis_id)
        if d.is_dir():
            shutil.rmtree(d)
            return True
        return False

    def read_png(self, analysis_id: str, stored_filename: str) -> np.ndarray:
        with Image.open(self.path_for(analysis_id, stored_filename)) as im:
            return np.array(im)
