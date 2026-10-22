"""Procedural SYNTHETIC before/after scenes with exact ground-truth masks.

These are NOT real satellite images. They exist so that the full pipeline
(alignment, detection, evaluation, training loop) can be exercised and unit-tested
offline without downloading a dataset. Everything produced here is labelled
"synthetic" in manifests, the UI and checkpoint metadata.

Each scene contains textured terrain, vegetation, a river, roads and buildings.
The AFTER image contains a scenario-specific change (urban growth, flooding,
vegetation clearing) plus nuisance differences that real imagery also has:
a small rotation/translation, illumination change and sensor noise.
"""
from __future__ import annotations

import cv2
import numpy as np

SCENARIOS = ("urban_expansion", "flood_extent", "vegetation_clearing")


def _smooth_noise(rng: np.random.Generator, size: int, scale: int) -> np.ndarray:
    small = rng.random((max(2, size // scale), max(2, size // scale))).astype(np.float32)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def _texture(rng, size, amp):
    return rng.normal(0, amp, (size, size, 1)).astype(np.float32)


def _draw_building(img, mask, rng, x, y, w, h):
    roof = rng.choice([(150, 150, 155), (185, 95, 70), (210, 205, 200), (120, 125, 135), (170, 160, 140)])
    cv2.rectangle(img, (x + 3, y + 3), (x + w + 3, y + h + 3), (45, 45, 50), -1)  # shadow
    cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in roof), -1)
    cv2.line(img, (x, y + h // 2), (x + w, y + h // 2), tuple(int(c * 0.8) for c in roof), 1)  # ridge
    if mask is not None:
        cv2.rectangle(mask, (x, y), (x + w + 3, y + h + 3), 1, -1)


def _base_scene(rng: np.random.Generator, size: int):
    terrain = _smooth_noise(rng, size, 64)[..., None]
    soil = np.array([150, 130, 95], np.float32)
    grass = np.array([105, 140, 75], np.float32)
    img = soil * (1 - terrain) + grass * terrain + _texture(rng, size, 6)

    veg = (_smooth_noise(rng, size, 48) > 0.62).astype(np.uint8)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    forest = np.array([45, 85, 40], np.float32) + _texture(rng, size, 10)
    img = np.where(veg[..., None] > 0, forest, img)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # river: smooth sinusoidal band, left to right
    xs = np.arange(size)
    yc = (size * 0.72 + size * 0.06 * np.sin(xs / size * 2 * np.pi * 1.3 + rng.random() * 6)).astype(np.int32)
    river = np.zeros((size, size), np.uint8)
    for x in xs:
        cv2.circle(river, (int(x), int(yc[x])), 9, 1, -1)
    img[river > 0] = (40, 70, 95)

    roads = np.zeros((size, size), np.uint8)
    cv2.line(roads, (0, size // 3), (size, size // 3 + 20), 1, 7)
    cv2.line(roads, (size // 4, 0), (size // 4 + 15, size), 1, 6)
    img[roads > 0] = (115, 115, 112)

    # town block in the north-west quadrant
    for _ in range(14):
        x = int(rng.integers(size // 4 + 15, size // 2 - 25))
        y = int(rng.integers(15, size // 3 - 30))
        _draw_building(img, None, rng, x, y, int(rng.integers(10, 22)), int(rng.integers(10, 22)))
    return img, {"vegetation": veg, "river": river, "roads": roads, "river_center": yc}


def _nuisance(img: np.ndarray, rng: np.random.Generator, shift=(5.0, -3.0), angle=1.2) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += shift
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = out.astype(np.float32) * 1.06 + np.array([4, 2, -3], np.float32)  # illumination / colour shift
    out += rng.normal(0, 3.0, out.shape)  # sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_pair(scenario: str, seed: int = 0, size: int = 512, misalign: bool = True):
    """Return (before_rgb, after_rgb, gt_mask_uint8{0,255}) - GT is in the BEFORE frame."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    before, layers = _base_scene(rng, size)
    after = before.copy()
    gt = np.zeros((size, size), np.uint8)

    if scenario == "urban_expansion":
        # new road + new buildings in the open area east of the town
        road = np.zeros_like(gt)
        cv2.line(road, (size // 2 + 10, size // 3 + 10), (size // 2 + 30, size // 2 + 60), 1, 7)
        after[road > 0] = (115, 115, 112)
        gt |= road
        for _ in range(int(rng.integers(7, 11))):
            x = int(rng.integers(size // 2 + 45, size - 40))
            y = int(rng.integers(30, size // 2 + 20))
            _draw_building(after, gt, rng, x, y, int(rng.integers(12, 26)), int(rng.integers(12, 26)))
    elif scenario == "flood_extent":
        flood = np.zeros_like(gt)
        yc = layers["river_center"]
        x0, x1 = size // 5, size * 4 // 5
        for x in range(x0, x1):
            width = int(14 + 38 * np.sin(np.pi * (x - x0) / (x1 - x0)))
            cv2.circle(flood, (x, int(yc[x])), width, 1, -1)
        flood &= (layers["river"] == 0).astype(np.uint8)
        water = np.array([55, 80, 100], np.float32) + _texture(rng, size, 3)
        after = np.where(flood[..., None] > 0, np.clip(water, 0, 255).astype(np.uint8), after)
        gt |= flood
    else:  # vegetation_clearing
        veg = layers["vegetation"]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(veg)
        if n > 1:
            order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
            cleared = np.isin(labels, order[:2]).astype(np.uint8)
        else:
            cleared = np.zeros_like(gt)
            cv2.circle(cleared, (size // 2, size // 2), size // 8, 1, -1)
        soil = np.array([165, 135, 100], np.float32) + _texture(rng, size, 7)
        after = np.where(cleared[..., None] > 0, np.clip(soil, 0, 255).astype(np.uint8), after)
        gt |= cleared

    if misalign:
        after = _nuisance(after, rng)
    else:
        after = np.clip(after.astype(np.float32) + rng.normal(0, 2.0, after.shape), 0, 255).astype(np.uint8)
    return before, after, (gt > 0).astype(np.uint8) * 255
