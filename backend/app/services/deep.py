"""Deep-learning inference service (FC-Siam-diff).

The service tries to load the checkpoint at ``ORBIT_MODEL_PATH`` on startup.
If it is missing or invalid, ``status`` says so and deep analysis requests are
rejected with HTTP 409 - the API never silently falls back to the classical
baseline while claiming a model produced the result.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

from app.models.checkpoint import CheckpointError, load_checkpoint

logger = logging.getLogger("orbit.deep")


def resolve_device(pref: str = "auto") -> str:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


class ModelService:
    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.metadata: dict = {}
        self.status = "not_loaded"  # loaded | missing | invalid | not_loaded
        self.message = "Model has not been loaded yet."
        self.device = "cpu"
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.status == "loaded"

    def load(self, path: Path, device_pref: str = "auto") -> None:
        self.device = resolve_device(device_pref)
        self.model, self.metadata = None, {}
        try:
            self.model, self.metadata = load_checkpoint(path, self.device)
            self.status = "loaded"
            self.message = f"Loaded {self.metadata.get('model_name', 'checkpoint')} on {self.device}."
            if str(self.metadata.get("dataset", "")).startswith("synthetic"):
                self.message += (" WARNING: this checkpoint was trained only on synthetic smoke-test data "
                                 "and is not meaningful for real imagery.")
            logger.info(self.message)
        except FileNotFoundError:
            self.status = "missing"
            self.message = ("No trained checkpoint found at ORBIT_MODEL_PATH. Deep-learning analysis is "
                            "unavailable; the classical baseline remains available. See docs/model_card.md "
                            "for how to train one.")
            logger.warning("Model checkpoint missing - deep-learning mode disabled.")
        except CheckpointError as exc:
            self.status = "invalid"
            self.message = f"Checkpoint rejected: {exc}"
            logger.error(self.message)

    def info(self) -> dict:
        meta = {k: v for k, v in self.metadata.items() if k not in ("training_args",)}
        return {"status": self.status, "ready": self.ready, "message": self.message,
                "device": self.device, "metadata": meta or None}

    def predict(self, before: np.ndarray, after: np.ndarray, tile: int = 256, overlap: int = 32,
                batch_size: int = 8) -> tuple[np.ndarray, float]:
        """Return (probability map float32 [0,1], inference_ms). Tiled sliding-window inference."""
        if not self.ready:
            raise RuntimeError("model not loaded")
        norm = self.metadata.get("normalization", {})
        mean = np.array(norm.get("mean", [0.5] * 3), dtype=np.float32)
        std = np.array(norm.get("std", [0.25] * 3), dtype=np.float32)
        stride_req = getattr(self.model, "stride", 16)

        h, w = before.shape[:2]
        # Pad so every tile is full-size and divisible by the network stride.
        tile = max(stride_req, (tile // stride_req) * stride_req)
        ph, pw = max(h, tile), max(w, tile)
        ph += (-ph) % stride_req
        pw += (-pw) % stride_req

        def prep(img: np.ndarray) -> np.ndarray:
            x = (img.astype(np.float32) / 255.0 - mean) / std
            pad_mode = "reflect" if (ph - h) < h and (pw - w) < w else "edge"
            x = np.pad(x, ((0, ph - h), (0, pw - w), (0, 0)), mode=pad_mode)
            return np.ascontiguousarray(x.transpose(2, 0, 1))

        xb, xa = prep(before), prep(after)
        step = max(1, tile - overlap)
        ys = list(range(0, max(ph - tile, 0) + 1, step))
        xs = list(range(0, max(pw - tile, 0) + 1, step))
        if ys[-1] != ph - tile:
            ys.append(ph - tile)
        if xs[-1] != pw - tile:
            xs.append(pw - tile)
        coords = [(y, x) for y in ys for x in xs]

        acc = np.zeros((ph, pw), dtype=np.float32)
        cnt = np.zeros((ph, pw), dtype=np.float32)
        t0 = time.perf_counter()
        with self._lock, torch.inference_mode():
            for i in range(0, len(coords), batch_size):
                chunk = coords[i:i + batch_size]
                tb = torch.from_numpy(np.stack([xb[:, y:y + tile, x:x + tile] for y, x in chunk])).to(self.device)
                ta = torch.from_numpy(np.stack([xa[:, y:y + tile, x:x + tile] for y, x in chunk])).to(self.device)
                prob = torch.sigmoid(self.model(tb, ta))[:, 0].float().cpu().numpy()
                for (y, x), p in zip(chunk, prob):
                    acc[y:y + tile, x:x + tile] += p
                    cnt[y:y + tile, x:x + tile] += 1
        ms = (time.perf_counter() - t0) * 1000
        return (acc / np.maximum(cnt, 1))[:h, :w].astype(np.float32), ms


model_service = ModelService()
