"""Paired change-detection dataset loader (LEVIR-CD layout).

Expected layout (identical to the official LEVIR-CD release):

    <root>/
      train/ A/*.png   B/*.png   label/*.png
      val/   A/*.png   B/*.png   label/*.png
      test/  A/*.png   B/*.png   label/*.png

A = earlier image, B = later image, label = binary mask (0 = no change, 255 = change).
Files are matched by identical file name across A, B and label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_pairs(root: Path, split: str) -> list[tuple[Path, Path, Path]]:
    base = Path(root) / split
    a_dir, b_dir, l_dir = base / "A", base / "B", base / "label"
    for d in (a_dir, b_dir, l_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"missing directory: {d.relative_to(Path(root).parent) if d.is_absolute() else d}")
    pairs = []
    for a in sorted(p for p in a_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        b, lab = b_dir / a.name, l_dir / a.name
        if not lab.exists():
            lab = l_dir / (a.stem + ".png")
        if b.exists() and lab.exists():
            pairs.append((a, b, lab))
    return pairs


def _read_rgb(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return np.array(im.convert("RGB"))


def _read_mask(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return (np.array(im.convert("L")) > 127).astype(np.float32)


class ChangeDetectionDataset(Dataset):
    """Yields (before[3,H,W], after[3,H,W], mask[1,H,W]) float tensors.

    train: `crops_per_image` random crops per image per epoch + flips/rotations.
    val/test: deterministic non-overlapping grid of crops covering each image.
    """

    def __init__(self, root: Path, split: str, patch_size: int = 256, augment: bool = False,
                 crops_per_image: int = 4, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25),
                 cache: bool = True, seed: int = 0):
        self.pairs = list_pairs(root, split)
        if not self.pairs:
            raise FileNotFoundError(f"no image pairs found for split '{split}'")
        self.patch, self.augment = patch_size, augment
        self.mean = np.array(mean, np.float32)
        self.std = np.array(std, np.float32)
        self.cache: dict[int, tuple] = {} if cache else None  # type: ignore[assignment]
        self.rng = np.random.default_rng(seed)
        if augment:
            self.index = [(i, None) for i in range(len(self.pairs)) for _ in range(crops_per_image)]
        else:
            self.index = []
            for i, (a, _, _) in enumerate(self.pairs):
                with Image.open(a) as im:
                    w, h = im.size
                ys = list(range(0, max(h - patch_size, 0) + 1, patch_size)) or [0]
                xs = list(range(0, max(w - patch_size, 0) + 1, patch_size)) or [0]
                self.index += [(i, (y, x)) for y in ys for x in xs]

    def __len__(self) -> int:
        return len(self.index)

    def _get(self, i: int):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        a, b, lab = self.pairs[i]
        item = (_read_rgb(a), _read_rgb(b), _read_mask(lab))
        if self.cache is not None:
            self.cache[i] = item
        return item

    def __getitem__(self, idx: int):
        i, pos = self.index[idx]
        a, b, m = self._get(i)
        h, w = m.shape
        p = self.patch
        if pos is None:
            y = int(self.rng.integers(0, max(h - p, 0) + 1))
            x = int(self.rng.integers(0, max(w - p, 0) + 1))
        else:
            y, x = pos
        a, b, m = a[y:y + p, x:x + p], b[y:y + p, x:x + p], m[y:y + p, x:x + p]
        if a.shape[0] < p or a.shape[1] < p:  # pad small images
            pad = ((0, p - a.shape[0]), (0, p - a.shape[1]))
            a = np.pad(a, pad + ((0, 0),), mode="reflect")
            b = np.pad(b, pad + ((0, 0),), mode="reflect")
            m = np.pad(m, pad, mode="constant")
        if self.augment:
            if self.rng.random() < 0.5:
                a, b, m = a[:, ::-1], b[:, ::-1], m[:, ::-1]
            if self.rng.random() < 0.5:
                a, b, m = a[::-1], b[::-1], m[::-1]
            k = int(self.rng.integers(0, 4))
            a, b, m = np.rot90(a, k), np.rot90(b, k), np.rot90(m, k)
            if self.rng.random() < 0.5:  # swap temporal order: change is symmetric
                a, b = b, a

        def to_t(x):
            x = (x.astype(np.float32) / 255.0 - self.mean) / self.std
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

        return to_t(a), to_t(b), torch.from_numpy(np.ascontiguousarray(m))[None]
