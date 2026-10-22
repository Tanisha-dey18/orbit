"""Change statistics and evaluation metrics."""
from __future__ import annotations

import cv2
import numpy as np


def connected_regions(mask: np.ndarray, intensity: np.ndarray | None = None, limit: int = 50) -> tuple[int, list[dict]]:
    """Return (region_count, regions sorted by area desc, truncated to `limit`)."""
    if not mask.any():
        return 0, []
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total = int(mask.sum())
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    regions = []
    for rank, lab in enumerate(order[:limit]):
        x, y, w, h, area = (int(v) for v in stats[lab])
        region = {
            "id": rank + 1,
            "label": int(lab),
            "bbox": [x, y, w, h],
            "area_px": area,
            "share_of_change": round(area / total, 4),
            "centroid": [round(float(centroids[lab][0]), 1), round(float(centroids[lab][1]), 1)],
        }
        if intensity is not None:
            region["mean_intensity"] = round(float(intensity[labels == lab].mean()), 4)
        regions.append(region)
    return n - 1, regions


def spatial_distribution(mask: np.ndarray, valid: np.ndarray, grid: int = 3) -> list[list[float]]:
    """Percentage of valid pixels changed in each cell of a grid x grid partition."""
    h, w = mask.shape
    out = []
    for i in range(grid):
        row = []
        for j in range(grid):
            ys, ye = i * h // grid, (i + 1) * h // grid
            xs, xe = j * w // grid, (j + 1) * w // grid
            v = valid[ys:ye, xs:xe].sum()
            row.append(round(float(mask[ys:ye, xs:xe].sum() / v * 100), 3) if v else 0.0)
        out.append(row)
    return out


def change_statistics(mask: np.ndarray, valid: np.ndarray, intensity: np.ndarray) -> dict:
    """changed_pixel_percentage = changed_pixel_count / valid_comparison_pixel_count * 100"""
    mask = mask & valid
    valid_count = int(valid.sum())
    changed = int(mask.sum())
    pct = changed / valid_count * 100 if valid_count else 0.0
    region_count, regions = connected_regions(mask, intensity)
    vals = intensity[valid]
    return {
        "valid_pixel_count": valid_count,
        "changed_pixel_count": changed,
        "changed_pixel_percentage": round(pct, 4),
        "total_pixel_count": int(valid.size),
        "region_count": region_count,
        "largest_region_px": regions[0]["area_px"] if regions else 0,
        "regions": regions,
        "mean_intensity_changed": round(float(intensity[mask].mean()), 4) if changed else 0.0,
        "mean_intensity_all": round(float(vals.mean()), 4) if vals.size else 0.0,
        "intensity_p95": round(float(np.percentile(vals, 95)), 4) if vals.size else 0.0,
        "spatial_grid_pct": spatial_distribution(mask, valid),
    }


def intensity_histogram(intensity: np.ndarray, valid: np.ndarray, bins: int = 32) -> list[dict]:
    hist, edges = np.histogram(intensity[valid], bins=bins, range=(0.0, 1.0))
    return [{"bin_start": round(float(edges[i]), 4), "count": int(hist[i])} for i in range(bins)]


def binary_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Pixel-level precision / recall / F1 / IoU for the 'change' class.

    Metrics whose denominator is zero are returned as None (undefined), not 0 or 1.
    """
    if valid is None:
        valid = np.ones_like(pred, dtype=bool)
    p, g = pred.astype(bool)[valid], gt.astype(bool)[valid]
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    return finalize_counts(tp, fp, fn, tn)


def finalize_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    def div(a, b):
        return round(a / b, 6) if b else None

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "iou": div(tp, tp + fp + fn),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
    }
