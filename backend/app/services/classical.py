"""Classical (non-learned) change detection - the BASELINE method.

These algorithms are deterministic image-processing operations, not trained AI.
They respond to *any* radiometric difference (lighting, season, shadows, sensor
noise) as well as to real change, which is exactly why a learned model is useful.

Methods
-------
grayscale_diff : |gray(after) - gray(before)| / 255
cva            : Change Vector Analysis - Euclidean distance between RGB vectors,
                 normalised by the maximum possible distance (255*sqrt(3)).
"""
from __future__ import annotations

import cv2
import numpy as np

CLASSICAL_METHODS = {
    "grayscale_diff": "Absolute grayscale difference",
    "cva": "Change Vector Analysis (RGB magnitude)",
}


def difference_intensity(before: np.ndarray, after: np.ndarray, method: str = "cva",
                         blur_sigma: float = 1.0) -> np.ndarray:
    """Return a float32 change-intensity map in [0, 1]."""
    if method not in CLASSICAL_METHODS:
        raise ValueError(f"Unknown classical method '{method}'. Options: {list(CLASSICAL_METHODS)}")
    b = before.astype(np.float32)
    a = after.astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        b = cv2.GaussianBlur(b, (0, 0), blur_sigma)
        a = cv2.GaussianBlur(a, (0, 0), blur_sigma)
    if method == "grayscale_diff":
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        intensity = np.abs((a * w).sum(-1) - (b * w).sum(-1)) / 255.0
    else:
        intensity = np.sqrt(((a - b) ** 2).sum(-1)) / (255.0 * np.sqrt(3.0))
    return np.clip(intensity, 0.0, 1.0).astype(np.float32)


def otsu_threshold(intensity: np.ndarray, valid: np.ndarray) -> float:
    """Otsu's method on valid pixels - picks the threshold that best separates two modes."""
    vals = (intensity[valid] * 255).astype(np.uint8)
    if vals.size == 0:
        return 0.5
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def postprocess_mask(intensity: np.ndarray, valid: np.ndarray, threshold: float,
                     morph_kernel: int = 3, min_region_px: int = 16) -> np.ndarray:
    """Threshold -> morphological opening/closing -> drop tiny connected components."""
    mask = ((intensity >= threshold) & valid).astype(np.uint8)
    if morph_kernel and morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)   # removes isolated speckle
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  # fills small holes
    if min_region_px and min_region_px > 1 and mask.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_region_px
        mask = keep[labels].astype(np.uint8)
    return (mask > 0) & valid
