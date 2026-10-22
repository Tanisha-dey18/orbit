"""Classical change detection, statistics, metrics, geo area and interpretation."""
import numpy as np
import pytest

from app.services import classical, metrics
from app.services.geo import GeoInfo, estimate_changed_area
from app.services.interpretation import interpret


def scene():
    rng = np.random.default_rng(1)
    img = np.full((100, 100, 3), 120, np.uint8) + rng.integers(0, 3, (100, 100, 3), dtype=np.uint8)
    return img


def test_identical_images_give_empty_mask():
    img = scene()
    inten = classical.difference_intensity(img, img.copy(), "cva")
    valid = np.ones(img.shape[:2], bool)
    mask = classical.postprocess_mask(inten, valid, 0.12)
    stats = metrics.change_statistics(mask, valid, inten)
    assert stats["changed_pixel_count"] == 0
    assert stats["changed_pixel_percentage"] == 0.0
    assert stats["region_count"] == 0 and stats["regions"] == []


@pytest.mark.parametrize("method", ["cva", "grayscale_diff"])
def test_known_square_change_percentage(method):
    a = scene()
    b = a.copy()
    b[20:40, 30:50] = 250  # 400 changed pixels
    valid = np.ones(a.shape[:2], bool)
    inten = classical.difference_intensity(a, b, method, blur_sigma=0)
    mask = classical.postprocess_mask(inten, valid, 0.2, morph_kernel=0, min_region_px=0)
    stats = metrics.change_statistics(mask, valid, inten)
    assert stats["changed_pixel_count"] == 400
    assert stats["changed_pixel_percentage"] == pytest.approx(4.0)
    assert stats["region_count"] == 1
    assert stats["regions"][0]["bbox"] == [30, 20, 20, 20]


def test_percentage_uses_valid_pixels_only():
    a = scene()
    b = a.copy()
    b[0:10, 0:10] = 250  # 100 px
    valid = np.zeros(a.shape[:2], bool)
    valid[:50] = True  # 5000 valid pixels
    inten = classical.difference_intensity(a, b, "cva", blur_sigma=0)
    mask = classical.postprocess_mask(inten, valid, 0.2, 0, 0)
    stats = metrics.change_statistics(mask, valid, inten)
    assert stats["valid_pixel_count"] == 5000
    assert stats["changed_pixel_percentage"] == pytest.approx(2.0)


def test_morphology_removes_speckle():
    a = scene()
    b = a.copy()
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(0, 100, 40), rng.integers(0, 100, 40)
    b[ys, xs] = 255  # isolated single-pixel noise
    valid = np.ones(a.shape[:2], bool)
    inten = classical.difference_intensity(a, b, "cva", blur_sigma=0)
    assert classical.postprocess_mask(inten, valid, 0.2, 0, 0).sum() > 0
    assert classical.postprocess_mask(inten, valid, 0.2, 3, 16).sum() == 0


def test_otsu_threshold_between_modes():
    inten = np.zeros((50, 50), np.float32)
    inten[:10] = 0.8
    t = classical.otsu_threshold(inten, np.ones_like(inten, bool))
    assert 0.0 <= t < 0.8


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        classical.difference_intensity(scene(), scene(), "magic")


def test_binary_metrics_known_values():
    pred = np.array([[1, 1, 0, 0]], bool)
    gt = np.array([[1, 0, 1, 0]], bool)
    m = metrics.binary_metrics(pred, gt)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["iou"] == pytest.approx(1 / 3, abs=1e-6)


def test_binary_metrics_undefined_when_empty():
    z = np.zeros((4, 4), bool)
    m = metrics.binary_metrics(z, z)
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None and m["iou"] is None
    assert m["accuracy"] == 1.0


def _geo(**kw):
    base = dict(crs_wkt="X", crs_name="EPSG:32643", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 1e6),
                width=100, height=100, is_projected=True, is_geographic=False, linear_unit_m=1.0)
    base.update(kw)
    return GeoInfo(**base)


def test_area_projected_crs():
    mask = np.zeros((100, 100), bool)
    mask[:10, :10] = True
    area, _ = estimate_changed_area(mask, _geo())
    assert area == pytest.approx(100 * 100.0)  # 100 px * 10m * 10m
    area2, _ = estimate_changed_area(mask, _geo(), 2.0, 2.0)  # downscaled 2x -> 4x area per pixel
    assert area2 == pytest.approx(4 * 100 * 100.0)


def test_area_geographic_crs_near_equator():
    mask = np.ones((10, 10), bool)
    g = _geo(is_projected=False, is_geographic=True, linear_unit_m=None,
             transform=(0.0001, 0.0, 0.0, 0.0, -0.0001, 0.0005))
    area, _ = estimate_changed_area(mask, g)
    # 0.001 deg x 0.001 deg near the equator ~ 111.2 m x 111.2 m
    assert area == pytest.approx(111.2 ** 2, rel=0.01)


def test_area_not_reported_without_metadata():
    area, why = estimate_changed_area(np.ones((5, 5), bool), None)
    assert area is None and "No georeferencing" in why
    area, _ = estimate_changed_area(np.ones((5, 5), bool), _geo(linear_unit_m=None))
    assert area is None


def test_interpretation_empty_and_context_independent(urban_pair):
    before, after, gt = urban_pair
    valid = np.ones(gt.shape, bool)
    empty = np.zeros(gt.shape, bool)
    stats = metrics.change_statistics(empty, valid, np.zeros(gt.shape, np.float32))
    res = interpret(before, before, empty, stats, 0.12)
    assert all(c["regions"] == 0 for c in res["categories"].values())
    assert any("No changed pixels" in s for s in res["summary"])

    mask = gt > 0
    inten = mask.astype(np.float32) * 0.5
    stats = metrics.change_statistics(mask, valid, inten)
    r1 = interpret(before, before, mask, stats, 0.12, context="general")
    r2 = interpret(before, before, mask, stats, 0.12, context="flood")
    assert r1["categories"] == r2["categories"]  # context never changes categories
    assert "Satellite imagery alone" in r1["notice"]


def test_vegetation_clearing_is_recognised():
    from app.utils.synthetic import generate_pair
    before, after, gt = generate_pair("vegetation_clearing", seed=5, size=256, misalign=False)
    mask = gt > 0
    valid = np.ones(mask.shape, bool)
    stats = metrics.change_statistics(mask, valid, mask.astype(np.float32))
    res = interpret(before, after, mask, stats, 0.12)
    assert res["categories"]["vegetation"]["share_of_change"] > 0.5
