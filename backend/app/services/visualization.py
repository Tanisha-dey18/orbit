"""Rendering of difference heatmaps, masks and overlays."""
from __future__ import annotations

import cv2
import numpy as np

OVERLAY_COLOR = (255, 64, 96)  # RGB - warm red-pink contrasts with most terrain
CONTOUR_COLOR = (0, 229, 255)  # cyan outlines


def heatmap(intensity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Colour-map the change intensity (TURBO: blue = low, red = high). Invalid pixels are black."""
    u8 = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def mask_image(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def intensity_image(intensity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.round(intensity * 255.0), 0, 255).astype(np.uint8)
    u8[~valid] = 0
    return u8


def overlay(base: np.ndarray, mask: np.ndarray, valid: np.ndarray, alpha: float = 0.5,
            draw_contours: bool = True) -> np.ndarray:
    out = base.astype(np.float32).copy()
    color = np.array(OVERLAY_COLOR, dtype=np.float32)
    out[mask] = out[mask] * (1 - alpha) + color * alpha
    out[~valid] *= 0.35  # dim pixels that were excluded from the comparison
    out = np.clip(out, 0, 255).astype(np.uint8)
    if draw_contours and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, CONTOUR_COLOR, 1)
    return out
