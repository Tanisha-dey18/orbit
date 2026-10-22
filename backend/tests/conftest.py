"""Test configuration: every test session gets its own temp DB / artifact dir / model path."""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_TMP = Path(tempfile.mkdtemp(prefix="orbit_test_"))
os.environ["ORBIT_DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["ORBIT_ARTIFACT_DIR"] = str(_TMP / "artifacts")
os.environ["ORBIT_MODEL_PATH"] = str(_TMP / "no_model_here.pt")
os.environ["ORBIT_MAX_UPLOAD_MB"] = "5"
os.environ["ORBIT_LOG_LEVEL"] = "WARNING"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.synthetic import generate_pair  # noqa: E402


def png_bytes(arr: np.ndarray, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def urban_pair():
    return generate_pair("urban_expansion", seed=11, size=256)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def textured_image():
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
    import cv2
    img = cv2.resize(base, (192, 192), interpolation=cv2.INTER_CUBIC)
    return img
