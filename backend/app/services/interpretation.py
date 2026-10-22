"""Disaster-intelligence module: conservative, rules-based interpretation.

IMPORTANT - what this module is and is not
------------------------------------------
* The change MASK comes from the detection method (classical baseline or model).
* The CATEGORY of each changed region here is a transparent RGB HEURISTIC, not a
  trained semantic-segmentation model. Every label is reported with the evidence
  that triggered it, and anything that does not clearly meet a rule is reported as
  "unclassified".
* The user-selected *context* only changes which findings the summary emphasises.
  It never changes the mask, the statistics or the per-region categories.
* RGB imagery alone usually cannot establish the cause or severity of a change
  (e.g. muddy floodwater looks like bare soil; shadows look like water).
"""
from __future__ import annotations

import cv2
import numpy as np

CATEGORIES = {
    "infrastructure": "Infrastructure change (linear feature, heuristic)",
    "building": "Building-related visual change (heuristic)",
    "vegetation": "Vegetation-related change (heuristic)",
    "flood": "Potential flood-related surface change (heuristic)",
    "unclassified": "Unclassified visual change",
}
CONTEXTS = ("general", "urban", "flood", "vegetation", "post_disaster")
_CONTEXT_FOCUS = {
    "urban": ["building", "infrastructure"],
    "flood": ["flood"],
    "vegetation": ["vegetation"],
    "post_disaster": ["building", "infrastructure", "flood"],
    "general": [],
}
MIN_CLASSIFY_PX = 24  # regions smaller than this are too small to characterise
NOTICE = ("Satellite imagery alone may be insufficient to determine the cause or severity of a change. "
          "Categories are rule-based RGB heuristics, not validated semantic predictions; confirm findings "
          "with ground truth, higher-resolution imagery or domain experts before acting on them.")


def _pixel_indicators(rgb: np.ndarray) -> dict[str, np.ndarray]:
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1e-6
    exg = (2 * g - r - b) / s  # excess-green on chromatic coordinates
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, val = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_std = np.sqrt(np.maximum(cv2.blur(gray ** 2, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2, 0))
    edges = cv2.Canny(rgb, 60, 150) > 0
    return {
        "vegetation": (exg > 0.08) & (g >= r) & (g >= b),
        "water": (b >= r) & (val < 0.55) & (local_std < 12) & (exg < 0.12),
        "built": (sat < 0.22) & (val > 0.4),
        "edges": edges,
    }


def _shape(region: np.ndarray) -> dict:
    ys, xs = np.nonzero(region)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    long_side, short_side = max(rw, rh, 1.0), max(min(rw, rh), 1.0)
    return {"elongation": round(long_side / short_side, 2),
            "extent": round(float(region.sum()) / (long_side * short_side), 3)}


def classify_region(region: np.ndarray, ind_b: dict, ind_a: dict) -> dict:
    n = int(region.sum())
    frac = {k: (float(ind_b[k][region].mean()), float(ind_a[k][region].mean())) for k in ind_b}
    shape = _shape(region)
    evidence: list[str] = []
    scores: dict[str, float] = {}

    veg_b, veg_a = frac["vegetation"]
    if abs(veg_b - veg_a) >= 0.4:
        scores["vegetation"] = abs(veg_b - veg_a)
        direction = "loss" if veg_b > veg_a else "gain"
        evidence.append(f"green-dominant pixels {veg_b:.0%} -> {veg_a:.0%} (possible vegetation {direction})")

    wat_b, wat_a = frac["water"]
    if wat_a - wat_b >= 0.4:
        scores["flood"] = wat_a - wat_b
        evidence.append(f"dark, smooth, blue-leaning (water-like) pixels {wat_b:.0%} -> {wat_a:.0%}")

    blt_b, blt_a = frac["built"]
    edge_a = frac["edges"][1]
    if abs(blt_a - blt_b) >= 0.3:
        built_delta = abs(blt_a - blt_b)
        verb = "appeared" if blt_a > blt_b else "disappeared"
        if shape["elongation"] >= 4.0 or shape["extent"] < 0.3:
            scores["infrastructure"] = built_delta
            evidence.append(f"grey/low-saturation surface {verb} in an elongated region "
                            f"(elongation {shape['elongation']}) - consistent with roads or linear structures")
        elif shape["extent"] >= 0.45 and (edge_a >= 0.04 or frac["edges"][0] >= 0.04):
            scores["building"] = built_delta
            evidence.append(f"grey/low-saturation compact surface {verb} (extent {shape['extent']}, "
                            f"edge density {edge_a:.2f}) - consistent with built structures")

    # Vegetation loss is often a side effect of construction or inundation; when another rule also
    # fires, prefer the more specific explanation (the vegetation evidence is still reported).
    if "vegetation" in scores and len(scores) > 1:
        scores["vegetation"] -= 0.2

    if n < MIN_CLASSIFY_PX or not scores:
        return {"category": "unclassified", "confidence": "n/a", "evidence": evidence or
                ["no heuristic rule was clearly satisfied"], "shape": shape}
    best = max(scores, key=scores.get)
    s = scores[best]
    return {"category": best, "confidence": "moderate" if s >= 0.6 else "low",
            "evidence": evidence, "shape": shape}


def _position_name(i: int, j: int) -> str:
    return [["north-west", "north", "north-east"], ["west", "centre", "east"],
            ["south-west", "south", "south-east"]][i][j]


def interpret(before: np.ndarray, after: np.ndarray, mask: np.ndarray, stats: dict,
              threshold: float, context: str = "general", reliable: bool = True,
              max_regions: int = 200) -> dict:
    if context not in CONTEXTS:
        context = "general"
    ind_b, ind_a = _pixel_indicators(before), _pixel_indicators(after)
    changed_total = int(mask.sum())
    summary_by_cat = {k: {"label": v, "regions": 0, "pixels": 0, "share_of_change": 0.0} for k, v in CATEGORIES.items()}
    regions_out: list[dict] = []

    if changed_total:
        n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        order = np.argsort(-cc_stats[1:, cv2.CC_STAT_AREA]) + 1
        for rank, lab in enumerate(order):
            area = int(cc_stats[lab, cv2.CC_STAT_AREA])
            if rank < max_regions:
                x, y, w, h = (int(v) for v in cc_stats[lab, :4])
                sub = labels[y:y + h, x:x + w] == lab
                crop = lambda d: {k: v[y:y + h, x:x + w] for k, v in d.items()}  # noqa: E731
                result = classify_region(sub, crop(ind_b), crop(ind_a))
                cat = result["category"]
                regions_out.append({"id": rank + 1, "bbox": [x, y, w, h], "area_px": area, **result})
            else:
                cat = "unclassified"
            summary_by_cat[cat]["regions"] += 1
            summary_by_cat[cat]["pixels"] += area
        for v in summary_by_cat.values():
            v["share_of_change"] = round(v["pixels"] / changed_total, 4)

    # ---- intensity / distribution descriptors ------------------------------------------
    mi = stats.get("mean_intensity_changed", 0.0)
    if not changed_total:
        intensity_level = "none"
    elif mi >= min(0.95, threshold * 2.0):
        intensity_level = "high"
    elif mi >= min(0.95, threshold * 1.4):
        intensity_level = "moderate"
    else:
        intensity_level = "low"
    grid = np.array(stats.get("spatial_grid_pct", [[0] * 3] * 3), dtype=float)
    if changed_total and grid.sum() > 0:
        i, j = np.unravel_index(int(grid.argmax()), grid.shape)
        dominant_share = grid.max() / grid.sum()
        distribution = (f"concentrated in the {_position_name(i, j)} of the scene" if dominant_share > 0.45
                        else "spread across multiple parts of the scene")
    else:
        distribution = "no change detected"

    # ---- areas requiring inspection: big & intense regions first ---------------------------
    inspect = sorted(stats.get("regions", []), key=lambda r: -r["area_px"] * r.get("mean_intensity", 1.0))[:5]
    inspection = [{"region_id": r["id"], "bbox": r["bbox"], "area_px": r["area_px"],
                   "reason": "large and high-intensity change" if r.get("mean_intensity", 0) >= threshold * 1.4
                   else "large changed region"} for r in inspect]

    # ---- natural-language summary -----------------------------------------------------------
    pct = stats.get("changed_pixel_percentage", 0.0)
    lines = []
    if not reliable:
        lines.append("Comparison flagged as UNRELIABLE (alignment or overlap problems) - treat all findings "
                     "below as provisional.")
    if not changed_total:
        lines.append("No changed pixels were detected at the selected threshold.")
    else:
        lines.append(f"{pct:.2f}% of valid pixels were classified as changed, forming "
                     f"{stats.get('region_count', 0)} connected region(s); change is {distribution}.")
        lines.append(f"Relative change intensity is {intensity_level} (mean {mi:.2f} vs threshold {threshold:.2f}).")
        for key in [k for k in CATEGORIES if k != "unclassified"]:
            c = summary_by_cat[key]
            if c["regions"]:
                lines.append(f"{c['label']}: {c['regions']} region(s), {c['share_of_change']:.0%} of changed pixels.")
        u = summary_by_cat["unclassified"]
        if u["regions"]:
            lines.append(f"Unclassified visual change: {u['regions']} region(s), {u['share_of_change']:.0%} of "
                         "changed pixels (no heuristic rule was clearly satisfied).")
    focus = _CONTEXT_FOCUS[context]
    if focus and changed_total:
        hits = [summary_by_cat[k] for k in focus if summary_by_cat[k]["regions"]]
        if hits:
            lines.append(f"Context '{context}': relevant heuristic categories were found - see category table. "
                         "These are indicators for further inspection, not confirmations.")
        else:
            lines.append(f"Context '{context}': no region met the heuristic rules for this context. The "
                         "selected context does not change the detection result.")

    return {
        "context": context,
        "categories": summary_by_cat,
        "regions": regions_out,
        "intensity_level": intensity_level,
        "spatial_distribution": distribution,
        "inspection_targets": inspection,
        "summary": lines,
        "notice": NOTICE,
        "method": "rgb_rules_v1",
    }
