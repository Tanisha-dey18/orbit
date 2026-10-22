"""Checkpoint format, saving and verified loading.

A PROJECT ORBIT checkpoint is a ``torch.save`` dict:

    {
      "format": "orbit-cd-v1",
      "arch": "fc_siam_diff",
      "arch_config": {"in_channels": 3, "base_channels": 16, "depth": 4},
      "state_dict": {...},
      "metadata": {
          "model_name": ..., "model_version": ...,
          "dataset": "LEVIR-CD" | "synthetic-smoke-test" | ...,
          "normalization": {"mean": [r,g,b], "std": [r,g,b]},   # on 0-1 scale
          "patch_size": 256, "epochs_trained": ..., "best_val": {...},
          "created_at": ..., "license_note": ..., "training_args": {...}
      }
    }

Loading uses ``weights_only=True`` (no arbitrary pickle code execution) and
``load_state_dict(strict=True)`` so an architecture/checkpoint mismatch is an error,
never a silent partial load.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.models.siamese_unet import ARCH_NAME, build_model, count_parameters

CHECKPOINT_FORMAT = "orbit-cd-v1"
DEFAULT_NORMALIZATION = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}


class CheckpointError(Exception):
    pass


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"created_at": datetime.now(timezone.utc).isoformat(), "normalization": DEFAULT_NORMALIZATION, **metadata}
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "arch": ARCH_NAME,
        "arch_config": model.arch_config(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": meta,
    }, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("checkpoint file not found")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"checkpoint could not be deserialised ({type(exc).__name__})") from exc
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"not a PROJECT ORBIT checkpoint (expected format '{CHECKPOINT_FORMAT}')")
    for key in ("arch", "arch_config", "state_dict", "metadata"):
        if key not in ckpt:
            raise CheckpointError(f"checkpoint is missing '{key}'")
    try:
        model = build_model(ckpt["arch"], ckpt["arch_config"])
    except (ValueError, TypeError, KeyError) as exc:
        raise CheckpointError(str(exc)) from exc
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError("checkpoint weights do not match the architecture: "
                              + str(exc).splitlines()[0][:300]) from exc
    model.to(device).eval()
    meta = dict(ckpt["metadata"])
    meta.setdefault("normalization", DEFAULT_NORMALIZATION)
    meta["arch"] = ckpt["arch"]
    meta["arch_config"] = ckpt["arch_config"]
    meta["parameters"] = count_parameters(model)
    meta["sha256"] = file_sha256(path)
    return model, meta
