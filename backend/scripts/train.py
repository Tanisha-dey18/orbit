"""Train FC-Siam-diff on a LEVIR-CD-style dataset.

Example (LEVIR-CD, after `prepare_dataset.py tile`):

    python scripts/train.py --data-root D:/datasets/LEVIR-CD-256 --dataset-name LEVIR-CD ^
        --epochs 100 --batch-size 16 --patch-size 256 --out ../data/checkpoints/orbit_fc_siam_diff.pt

Smoke test (synthetic data, a couple of minutes on CPU - NOT a usable model):

    python scripts/prepare_dataset.py synthetic --out ../data/synthetic_cd
    python scripts/train.py --data-root ../data/synthetic_cd --dataset-name synthetic-smoke-test ^
        --epochs 2 --out ../data/checkpoints/smoke_test.pt

The best checkpoint (by validation F1) is written to --out; the final epoch to <out>.last.pt;
the per-epoch history to <out>.history.json. Metrics stored in the checkpoint are the
actual validation results computed by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.checkpoint import DEFAULT_NORMALIZATION, save_checkpoint  # noqa: E402
from app.models.dataset import ChangeDetectionDataset  # noqa: E402
from app.models.siamese_unet import FCSiamDiff, count_parameters  # noqa: E402
from app.services.deep import resolve_device  # noqa: E402
from app.services.metrics import finalize_counts  # noqa: E402


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()


@torch.no_grad()
def validate(model, loader, device, threshold=0.5) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    loss_sum, n = 0.0, 0
    for a, b, m in loader:
        a, b, m = a.to(device), b.to(device), m.to(device)
        logits = model(a, b)
        loss_sum += F.binary_cross_entropy_with_logits(logits, m).item() * a.size(0)
        n += a.size(0)
        pred = torch.sigmoid(logits) >= threshold
        gt = m > 0.5
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
        tn += int((~pred & ~gt).sum())
    out = finalize_counts(tp, fp, fn, tn)
    out["loss"] = round(loss_sum / max(n, 1), 5)
    out["threshold"] = threshold
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset-name", required=True, help="Recorded in checkpoint metadata, e.g. LEVIR-CD")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--crops-per-image", type=int, default=1)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--pos-weight", type=float, default=2.0, help="BCE weight for the (rare) change class")
    ap.add_argument("--num-workers", type=int, default=0, help="Keep 0 on Windows unless you need speed")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train-batches", type=int, default=0, help="Limit batches/epoch (0 = all)")
    ap.add_argument("--model-version", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    norm = DEFAULT_NORMALIZATION
    train_ds = ChangeDetectionDataset(args.data_root, "train", args.patch_size, augment=True,
                                      crops_per_image=args.crops_per_image, mean=norm["mean"], std=norm["std"],
                                      seed=args.seed)
    val_ds = ChangeDetectionDataset(args.data_root, "val", args.patch_size, augment=False,
                                    mean=norm["mean"], std=norm["std"])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FCSiamDiff(base_channels=args.base_channels, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    pos_weight = torch.tensor([args.pos_weight], device=device)
    print(f"device={device} params={count_parameters(model):,} train_pairs={len(train_ds.pairs)} "
          f"train_crops/epoch={len(train_ds)} val_crops={len(val_ds)}")

    history, best_f1, best = [], -1.0, None
    version = args.model_version or time.strftime("%Y%m%d-%H%M%S")
    base_meta = {
        "model_name": "orbit-fc-siam-diff", "model_version": version, "dataset": args.dataset_name,
        "normalization": norm, "patch_size": args.patch_size,
        "training_args": {k: str(v) for k, v in vars(args).items()},
        "license_note": "Weights inherit the license/terms of the training dataset.",
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for bi, (a, b, m) in enumerate(train_dl):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            a, b, m = a.to(device), b.to(device), m.to(device)
            logits = model(a, b)
            loss = F.binary_cross_entropy_with_logits(logits, m, pos_weight=pos_weight) + dice_loss(logits, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * a.size(0)
            n += a.size(0)
        sched.step()
        val = validate(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": round(loss_sum / max(n, 1), 5), "val": val,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec))
        f1 = val["f1"] or 0.0
        if f1 > best_f1:
            best_f1, best = f1, {"epoch": epoch, **val}
            save_checkpoint(args.out, model, {**base_meta, "epochs_trained": epoch, "best_val": best})

    last = args.out.with_suffix(".last.pt")
    save_checkpoint(last, model, {**base_meta, "epochs_trained": args.epochs, "best_val": best,
                                  "final_val": history[-1]["val"] if history else None})
    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best val F1={best_f1:.4f} (epoch {best['epoch'] if best else '-'}) -> {args.out}")


if __name__ == "__main__":
    main()
