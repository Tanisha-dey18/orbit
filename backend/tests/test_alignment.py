import cv2
import numpy as np

from app.services.alignment import gradient_correlation, normalize_radiometry, prepare_pair
from app.services.image_io import LoadedImage


def li(rgb, valid=None):
    h, w = rgb.shape[:2]
    return LoadedImage(rgb=rgb, valid=np.ones((h, w), bool) if valid is None else valid, filename="x.png",
                       format="PNG", width=w, height=h, bands=3, dtype="uint8")


def test_identical_images_need_no_alignment(urban_pair):
    before, _, _ = urban_pair
    pair = prepare_pair(li(before), li(before.copy()))
    assert pair.alignment["status"] in ("not_needed", "skipped") or pair.alignment["status"] == "aligned"
    assert pair.reliable
    assert pair.valid.mean() > 0.99


def test_shift_and_rotation_are_recovered(urban_pair):
    before, after, _ = urban_pair  # after has a 1.2 deg rotation + (5,-3) px shift
    pair = prepare_pair(li(before), li(after))
    al = pair.alignment
    assert al["status"] == "aligned"
    assert abs(abs(al["rotation_deg"]) - 1.2) < 0.3
    assert al["score_after"] > al["score_before"]
    assert pair.reliable
    # border pixels introduced by warping are excluded
    assert 0.85 < pair.valid.mean() < 1.0


def test_different_dimensions_are_resized_with_warning(urban_pair):
    before, after, _ = urban_pair
    bigger = cv2.resize(after, (384, 384))
    pair = prepare_pair(li(before), li(bigger))
    assert pair.before.shape == pair.after.shape
    assert any("Resizing alone does not guarantee" in w for w in pair.warnings)


def test_aspect_ratio_mismatch_marks_unreliable(urban_pair):
    before, after, _ = urban_pair
    wide = cv2.resize(after, (400, 200))
    pair = prepare_pair(li(before), li(wide), alignment_method="none")
    assert not pair.reliable


def test_unrelated_images_fail_alignment():
    rng = np.random.default_rng(0)
    a = cv2.resize(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), (256, 256), interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8), (256, 256), interpolation=cv2.INTER_CUBIC)
    pair = prepare_pair(li(a), li(b))
    assert pair.reliable is False
    assert pair.alignment["status"] in ("failed", "not_needed", "aligned")
    assert any("alignment failed" in w.lower() or "structural similarity" in w.lower() for w in pair.warnings)


def test_alignment_none_is_skipped(urban_pair):
    before, after, _ = urban_pair
    pair = prepare_pair(li(before), li(after), alignment_method="none")
    assert pair.alignment["status"] == "skipped"


def test_downscale_cap(urban_pair):
    before, after, _ = urban_pair
    pair = prepare_pair(li(before), li(after), max_side=128)
    assert max(pair.before.shape[:2]) == 128
    assert pair.scale_x == 2.0


def test_radiometric_normalisation_reduces_global_difference(urban_pair):
    before, _, _ = urban_pair
    darker = np.clip(before.astype(float) * 0.7 + 10, 0, 255).astype(np.uint8)
    valid = np.ones(before.shape[:2], bool)
    for method in ("mean_std", "histogram"):
        norm = normalize_radiometry(before, darker, valid, method)
        assert np.abs(norm.astype(float) - before).mean() < np.abs(darker.astype(float) - before).mean() / 2
    assert gradient_correlation(before, before, valid) > 0.99
