"""Evaluate change detection against ground-truth masks.

Deep model:
    python scripts/evaluate.py --data-root D:/datasets/LEVIR-CD --split test ^
        --method deep --checkpoint ../data/checkpoints/orbit_fc_siam_diff.pt --threshold 0.5

Classical baseline (for comparison):
    python scripts/evaluate.py --data-root D:/datasets/LEVIR-CD --split test --method classical ^
        --classical-method cva --threshold 0.12

Bundled synthetic samples (data/samples/manifest.json):
    python scripts/evaluate.py --samples --method classical

Metrics are pixel-level for the "change" class. Global metrics are computed from
confusion counts accumulated over all images (the usual convention for LEVIR-CD);
the mean per-image F1 is also reported. Results are written to --out as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import REPO_ROOT  # noqa: E402
from app.models.dataset import list_pairs  # noqa: E402
from app.services import classical  # noqa: E402
from app.services.alignment import prepare_pair  # noqa: E402
from app.services.deep import ModelService  # noqa: E402
from app.services.image_io import load_image  # noqa: E402
from app.services.metrics import binary_metrics, finalize_counts  # noqa: E402


def _rgb(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.array(im.convert("RGB"))


def _mask(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.array(im.convert("L")) > 127


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", action="store_true", help="Evaluate on bundled synthetic samples (with alignment)")
    ap.add_argument("--method", choices=["deep", "classical"], required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--classical-method", default="cva", choices=list(classical.CLASSICAL_METHODS))
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--morph-kernel", type=int, default=3)
    ap.add_argument("--min-region-px", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N pairs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    threshold = args.threshold if args.threshold is not None else (0.5 if args.method == "deep" else 0.12)

    if args.samples:
        manifest = json.loads((REPO_ROOT / "data/samples/manifest.json").read_text(encoding="utf-8"))
        base = REPO_ROOT / "data/samples"
        pairs = [(base / s["files"]["before"], base / s["files"]["after"], base / s["files"]["ground_truth"])
                 for s in manifest["samples"]]
        dataset_label = "PROJECT ORBIT synthetic samples"
    else:
        if not args.data_root:
            ap.error("--data-root is required unless --samples is given")
        pairs = list_pairs(args.data_root, args.split)
        dataset_label = f"{args.data_root.name}/{args.split}"
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        sys.exit("No pairs found.")

    svc = None
    if args.method == "deep":
        if not args.checkpoint:
            ap.error("--checkpoint is required for --method deep")
        svc = ModelService()
        svc.load(args.checkpoint, args.device)
        if not svc.ready:
            sys.exit(f"Model not usable: {svc.message}")

    tp = fp = fn = tn = 0
    per_image, times = [], []
    for a_p, b_p, l_p in pairs:
        gt = _mask(l_p)
        if args.samples:  # bundled samples contain deliberate misalignment -> run the full pipeline prep
            pair = prepare_pair(load_image(a_p.read_bytes(), a_p.name), load_image(b_p.read_bytes(), b_p.name))
            before, after, valid = pair.before, pair.after, pair.valid
        else:  # benchmark datasets are distributed co-registered
            before, after = _rgb(a_p), _rgb(b_p)
            valid = np.ones(gt.shape, bool)
        t0 = time.perf_counter()
        if svc:
            intensity, _ = svc.predict(before, after)
        else:
            intensity = classical.difference_intensity(before, after, args.classical_method)
        pred = classical.postprocess_mask(intensity, valid, threshold, args.morph_kernel, args.min_region_px) \
            if args.method == "classical" else ((intensity >= threshold) & valid)
        times.append((time.perf_counter() - t0) * 1000)
        m = binary_metrics(pred, gt, valid)
        tp, fp, fn, tn = tp + m["tp"], fp + m["fp"], fn + m["fn"], tn + m["tn"]
        per_image.append({"file": a_p.name, "f1": m["f1"], "iou": m["iou"]})

    f1s = [p["f1"] for p in per_image if p["f1"] is not None]
    result = {
        "dataset": dataset_label,
        "method": args.method if args.method == "deep" else f"classical:{args.classical_method}",
        "checkpoint": {"model_version": svc.metadata.get("model_version"), "dataset": svc.metadata.get("dataset"),
                       "sha256": svc.metadata.get("sha256")} if svc else None,
        "threshold": threshold,
        "num_samples": len(pairs),
        "global": finalize_counts(tp, fp, fn, tn),
        "mean_per_image_f1": round(float(np.mean(f1s)), 6) if f1s else None,
        "mean_inference_ms": round(float(np.mean(times)), 2),
        "per_image": per_image,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
