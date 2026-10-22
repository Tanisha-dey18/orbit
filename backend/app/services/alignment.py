"""Image registration (alignment) and radiometric normalisation.

Change detection compares pixels at the same (row, col). If the AFTER image is
shifted by even a few pixels, every edge in the scene shows up as "change".
This module therefore:

* reprojects georeferenced pairs onto a common CRS/grid (when Rasterio is available),
* resizes mismatched non-georeferenced pairs (with an explicit warning - resizing
  does NOT make images geographically aligned),
* estimates a similarity transform with ORB features + RANSAC, falling back to
  OpenCV's ECC (enhanced correlation coefficient) maximisation,
* rejects transforms that are implausible or that make the match *worse*,
* tracks which pixels are valid after warping so borders are excluded from stats.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services.geo import GeoInfo, reproject_to_reference, same_grid
from app.services.image_io import LoadedImage

logger = logging.getLogger("orbit.alignment")

ALIGNMENT_METHODS = ("auto", "feature", "ecc", "none")
NORMALIZATION_METHODS = ("mean_std", "histogram", "none")


@dataclass
class PreparedPair:
    before: np.ndarray  # (H, W, 3) uint8
    after: np.ndarray  # (H, W, 3) uint8, aligned to `before`
    valid: np.ndarray  # (H, W) bool
    geo: GeoInfo | None
    scale_x: float  # original reference pixels per processed pixel
    scale_y: float
    alignment: dict
    warnings: list[str] = field(default_factory=list)
    reliable: bool = True


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gradient_correlation(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    """Normalised cross-correlation of gradient magnitudes inside `valid`.

    Gradients are fairly robust to illumination/seasonal colour changes, so this is a
    reasonable proxy for "are the structures in the same place".
    """
    def grad(x):
        g = cv2.GaussianBlur(_gray(x).astype(np.float32), (0, 0), 1.0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    ga, gb = grad(a)[valid], grad(b)[valid]
    if ga.size < 100:
        return 0.0
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    denom = float(np.sqrt((ga * ga).sum() * (gb * gb).sum()))
    return float((ga * gb).sum() / denom) if denom > 0 else 0.0


def _decompose(m: np.ndarray) -> tuple[float, float, float, float]:
    """Return (tx, ty, rotation_deg, scale) of a 2x3 similarity matrix."""
    scale = math.hypot(m[0, 0], m[1, 0])
    rot = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    return float(m[0, 2]), float(m[1, 2]), rot, scale


def _plausible(m: np.ndarray, shape: tuple[int, int]) -> bool:
    tx, ty, rot, scale = _decompose(m)
    h, w = shape
    return 0.8 <= scale <= 1.25 and abs(rot) <= 30 and abs(tx) <= 0.3 * w and abs(ty) <= 0.3 * h


def estimate_feature_transform(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """ORB keypoints + ratio-test matching + RANSAC similarity (rotation, scale, shift)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1, g2 = clahe.apply(_gray(ref)), clahe.apply(_gray(mov))
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    info = {"keypoints_before": len(k1), "keypoints_after": len(k2), "matches": 0, "inliers": 0}
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        info["reason"] = "too few distinctive keypoints"
        return None, info
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(d2, d1, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    info["matches"] = len(good)
    if len(good) < 12:
        info["reason"] = "too few reliable feature matches"
        return None, info
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    m, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                             maxIters=5000, confidence=0.995)
    if m is None or inliers is None:
        info["reason"] = "RANSAC could not fit a transform"
        return None, info
    n_in = int(inliers.sum())
    info["inliers"] = n_in
    info["inlier_ratio"] = round(n_in / len(good), 3)
    if n_in < 10 or n_in / len(good) < 0.2:
        info["reason"] = "inlier support too weak"
        return None, info
    return m.astype(np.float64), info


def estimate_ecc_transform(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Intensity-based Euclidean registration (cv2.findTransformECC) on a downsampled pyramid."""
    g1 = _gray(ref).astype(np.float32) / 255.0
    g2 = _gray(mov).astype(np.float32) / 255.0
    longest = max(g1.shape)
    factor = min(1.0, 768 / longest)
    if factor < 1.0:
        size = (int(g1.shape[1] * factor), int(g1.shape[0] * factor))
        g1s, g2s = cv2.resize(g1, size, interpolation=cv2.INTER_AREA), cv2.resize(g2, size, interpolation=cv2.INTER_AREA)
    else:
        g1s, g2s = g1, g2
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(g1s, g2s, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error:
        return None, {"reason": "ECC optimisation did not converge"}
    warp = warp.astype(np.float64)
    warp[:, 2] /= factor
    # ECC returns the map ref->mov; invert to get mov->ref like the feature path.
    inv = cv2.invertAffineTransform(warp)
    return inv, {"ecc_correlation": round(float(cc), 4)}


def normalize_radiometry(before: np.ndarray, after: np.ndarray, valid: np.ndarray, method: str) -> np.ndarray:
    """Match AFTER's colour distribution to BEFORE (reduces false change from lighting/sensor)."""
    if method == "none" or not valid.any():
        return after
    out = after.copy()
    for c in range(3):
        a_vals = after[..., c][valid].astype(np.float64)
        b_vals = before[..., c][valid].astype(np.float64)
        if method == "mean_std":
            sa, sb = a_vals.std(), b_vals.std()
            gain = sb / sa if sa > 1e-6 else 1.0
            mapped = (after[..., c].astype(np.float64) - a_vals.mean()) * gain + b_vals.mean()
            out[..., c] = np.clip(mapped, 0, 255).astype(np.uint8)
        elif method == "histogram":
            a_hist = np.bincount(a_vals.astype(np.uint8), minlength=256).astype(np.float64)
            b_hist = np.bincount(b_vals.astype(np.uint8), minlength=256).astype(np.float64)
            a_cdf = np.cumsum(a_hist) / a_hist.sum()
            b_cdf = np.cumsum(b_hist) / b_hist.sum()
            lut = np.interp(a_cdf, b_cdf, np.arange(256)).astype(np.uint8)
            out[..., c] = lut[after[..., c]]
        else:
            raise ValueError(f"Unknown normalisation method '{method}'")
    return out


def _downscale(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def prepare_pair(
    before: LoadedImage,
    after: LoadedImage,
    alignment_method: str = "auto",
    normalization: str = "mean_std",
    max_side: int = 2048,
) -> PreparedPair:
    if alignment_method not in ALIGNMENT_METHODS:
        raise ValueError(f"alignment_method must be one of {ALIGNMENT_METHODS}")
    warnings: list[str] = []
    reliable = True
    b_rgb, b_valid = before.rgb, before.valid.copy()
    a_rgb, a_valid = after.rgb, after.valid.copy()
    geo = before.geo
    report: dict = {"requested": alignment_method, "geospatial": "not_applicable"}

    # 1. Georeferenced pair -> common grid --------------------------------------------
    if before.geo and after.geo:
        if same_grid(before.geo, after.geo):
            report["geospatial"] = "same_grid"
        else:
            a_rgb, a_valid = reproject_to_reference(a_rgb, a_valid, after.geo, before.geo)
            report["geospatial"] = "reprojected_to_before_grid"
            warnings.append("AFTER image was reprojected onto the BEFORE image's CRS and pixel grid.")
            if a_valid.mean() < 0.05:
                warnings.append("The georeferenced images barely overlap; a reliable comparison is not possible.")
                reliable = False
    elif before.geo or after.geo:
        warnings.append("Only one image is georeferenced; geospatial reprojection was not possible "
                        "and physical area is reported only if the BEFORE image carries a CRS.")
        geo = before.geo

    # 2. Dimension mismatch -------------------------------------------------------------
    bh, bw = b_rgb.shape[:2]
    ah, aw = a_rgb.shape[:2]
    if (ah, aw) != (bh, bw):
        ar_b, ar_a = bw / bh, aw / ah
        a_rgb = cv2.resize(a_rgb, (bw, bh), interpolation=cv2.INTER_AREA if aw * ah > bw * bh else cv2.INTER_LINEAR)
        a_valid = cv2.resize(a_valid.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
        warnings.append(
            f"Image sizes differ (BEFORE {bw}x{bh}, AFTER {aw}x{ah}); AFTER was resized. "
            "Resizing alone does not guarantee geographic alignment."
        )
        if abs(ar_a - ar_b) / ar_b > 0.05:
            warnings.append("Aspect ratios differ by more than 5%; the images may cover different areas.")
            reliable = False

    # 3. Processing-size cap ------------------------------------------------------------
    scale = 1.0
    if max(bh, bw) > max_side:
        scale = max(bh, bw) / max_side
        size = (max(1, round(bw / scale)), max(1, round(bh / scale)))
        b_rgb, a_rgb = _downscale(b_rgb, size), _downscale(a_rgb, size)
        b_valid = cv2.resize(b_valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        a_valid = cv2.resize(a_valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        warnings.append(f"Images were downscaled by {scale:.2f}x to {size[0]}x{size[1]} for processing.")
    h, w = b_rgb.shape[:2]
    scale_x = bw / w
    scale_y = bh / h

    # 4. Pixel-level registration -------------------------------------------------------
    base_valid = b_valid & a_valid
    score_before = gradient_correlation(b_rgb, a_rgb, base_valid)
    report.update({"method_used": "none", "status": "skipped", "score_before": round(score_before, 4)})
    matrix = None
    if alignment_method != "none":
        candidates: list[tuple[str, np.ndarray, dict]] = []
        if alignment_method in ("auto", "feature"):
            m, info = estimate_feature_transform(b_rgb, a_rgb)
            report["feature"] = info
            if m is not None:
                candidates.append(("feature", m, info))
        if alignment_method == "ecc" or (alignment_method == "auto" and not candidates):
            m, info = estimate_ecc_transform(b_rgb, a_rgb)
            report["ecc"] = info
            if m is not None:
                candidates.append(("ecc", m, info))

        accepted = False
        for name, m, _ in candidates:
            if not _plausible(m, (h, w)):
                report.setdefault("rejected", []).append({"method": name, "reason": "implausible transform"})
                continue
            tx, ty, rot, sc = _decompose(m)
            if abs(tx) < 0.5 and abs(ty) < 0.5 and abs(rot) < 0.05 and abs(sc - 1) < 1e-3:
                report.update({"method_used": name, "status": "not_needed"})
                accepted = True
                break
            warped = cv2.warpAffine(a_rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            wvalid = cv2.warpAffine(a_valid.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
            wvalid = cv2.erode(wvalid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            score_after = gradient_correlation(b_rgb, warped, b_valid & wvalid)
            if score_after + 0.01 < score_before:
                report.setdefault("rejected", []).append(
                    {"method": name, "reason": "transform reduced structural similarity",
                     "score": round(score_after, 4)})
                continue
            a_rgb, a_valid, matrix = warped, wvalid, m
            report.update({
                "method_used": name, "status": "aligned", "score_after": round(score_after, 4),
                "translation_px": [round(tx, 2), round(ty, 2)], "rotation_deg": round(rot, 3),
                "scale": round(sc, 4),
            })
            accepted = True
            break
        if not accepted:
            report["status"] = "failed"
            reliable = False
            warnings.append(
                "Automatic alignment failed - the images could not be registered reliably. Detected "
                "changes may be caused by misalignment rather than real change on the ground."
            )

    valid = b_valid & a_valid
    report["matrix"] = matrix.round(6).tolist() if matrix is not None else None
    report["valid_fraction"] = round(float(valid.mean()), 4)
    if report.get("status") in ("skipped", "not_needed"):
        report["score_after"] = round(score_before, 4)
    final_score = report.get("score_after", score_before)
    if final_score < 0.15 and reliable:
        warnings.append(
            f"Low structural similarity between images (gradient correlation {final_score:.2f}). "
            "They may show different areas, or have very different viewing conditions."
        )
        reliable = False
    if valid.mean() < 0.5:
        warnings.append(f"Only {valid.mean() * 100:.0f}% of pixels are comparable after alignment.")
        reliable = False
    if valid.sum() < 1024:
        raise ValueError("Fewer than 1024 valid overlapping pixels - comparison is not possible.")

    a_rgb = normalize_radiometry(b_rgb, a_rgb, valid, normalization)
    a_rgb[~valid] = 0
    report["normalization"] = normalization

    return PreparedPair(before=b_rgb, after=a_rgb, valid=valid, geo=geo, scale_x=scale_x, scale_y=scale_y,
                        alignment=report, warnings=warnings, reliable=reliable)
