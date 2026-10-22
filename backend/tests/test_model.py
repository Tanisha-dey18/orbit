"""Siamese U-Net architecture, checkpoint loading and the model service."""
import numpy as np
import pytest
import torch

from app.models.checkpoint import CHECKPOINT_FORMAT, CheckpointError, load_checkpoint, save_checkpoint
from app.models.siamese_unet import FCSiamDiff, build_model
from app.services.deep import ModelService


def test_forward_shape():
    m = FCSiamDiff(base_channels=8).eval()
    x = torch.randn(2, 3, 64, 64)
    assert m(x, x).shape == (2, 1, 64, 64)


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError):
        build_model("resnet", {})


def test_checkpoint_roundtrip(tmp_path):
    m = FCSiamDiff(base_channels=8)
    p = tmp_path / "ok.pt"
    save_checkpoint(p, m, {"model_name": "t", "model_version": "1", "dataset": "unit-test"})
    loaded, meta = load_checkpoint(p)
    assert meta["dataset"] == "unit-test" and meta["parameters"] > 0 and len(meta["sha256"]) == 64
    for k, v in m.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_mismatched_weights_rejected(tmp_path):
    m = FCSiamDiff(base_channels=8)
    p = tmp_path / "bad.pt"
    save_checkpoint(p, m, {"dataset": "unit-test"})
    ck = torch.load(p, weights_only=True)
    ck["arch_config"]["base_channels"] = 16  # architecture no longer matches the weights
    torch.save(ck, p)
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(p)


def test_foreign_file_rejected(tmp_path):
    p = tmp_path / "foreign.pt"
    torch.save({"weights": torch.zeros(3)}, p)
    with pytest.raises(CheckpointError):
        load_checkpoint(p)
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(p)


def test_service_missing_and_invalid(tmp_path):
    svc = ModelService()
    svc.load(tmp_path / "missing.pt")
    assert svc.status == "missing" and not svc.ready
    with pytest.raises(RuntimeError):
        svc.predict(np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8))
    bad = tmp_path / "bad.pt"
    torch.save({"format": CHECKPOINT_FORMAT}, bad)
    svc.load(bad)
    assert svc.status == "invalid" and not svc.ready


@pytest.mark.parametrize("shape", [(64, 64), (100, 300), (300, 270)])
def test_service_predict_any_size(tmp_path, shape):
    p = tmp_path / "m.pt"
    save_checkpoint(p, FCSiamDiff(base_channels=8), {"dataset": "unit-test"})
    svc = ModelService()
    svc.load(p, "cpu")
    assert svc.ready
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (*shape, 3), dtype=np.uint8)
    prob, ms = svc.predict(a, a, tile=128, overlap=16)
    assert prob.shape == shape and prob.dtype == np.float32
    assert 0.0 <= prob.min() and prob.max() <= 1.0 and ms > 0
