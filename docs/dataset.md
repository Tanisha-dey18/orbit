# Dataset guide

## Candidates considered

| Dataset | Imagery | Size | Labels | Suitability |
|---|---|---|---|---|
| **LEVIR-CD** *(primary)* | Google Earth RGB, 0.5 m/px, Texas (USA), 2002–2018 | 637 pairs, 1024×1024 | binary building change (~31 k change instances) | Standard benchmark, clean official split, manageable size → **chosen** |
| WHU Building CD | aerial RGB, ~0.3 m/px (resampled), Christchurch NZ, 2012 vs 2016 | one large pair (32 507 × 15 354) | binary building change | good, but no official tiling/split; must be cropped yourself |
| S2Looking | side-looking satellite RGB, 0.5–0.8 m/px, worldwide rural areas | 5 000 pairs, 1024×1024 | building appeared / disappeared | harder (off-nadir); larger download |

All three are about **buildings**. None of them labels floods, vegetation or damage. That is why the "disaster
intelligence" categories in this project are heuristics and are labelled as such.

## LEVIR-CD (primary)

* **Reference:** H. Chen and Z. Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote
  Sensing Image Change Detection", *Remote Sensing*, 12(10):1662, 2020.
* **Access:** download it manually from the authors' project page (search for "LEVIR-CD dataset"; it is hosted by
  the LEVIR lab at Beihang University, with mirrors linked from that page). There is no automated download in this
  repository: the files sit behind cloud-drive links, and you should read and accept the terms yourself.
* **Licensing considerations:** the authors provide the dataset for **academic research**. The images come from Google
  Earth, so Google's terms also apply. Do **not** redistribute the images in this repository. Check the terms before
  publishing derived weights or using them commercially. This repository bundles no LEVIR-CD data.
* **Structure** (as distributed, and as expected by `app/models/dataset.py`):

  ```
  LEVIR-CD/
    train/ A/ B/ label/     445 pairs
    val/   A/ B/ label/      64 pairs
    test/  A/ B/ label/     128 pairs
  ```
  `A` = earlier image, `B` = later image, and `label` = an 8-bit PNG where 0 means no change and 255 means change. Files
  are matched by identical names.
* **Dimensions:** 1024×1024×3 (uint8).
* **Class balance:** change pixels are a small minority (a few percent). This is why `train.py` uses a positive-class
  BCE weight together with Dice loss.

### Preparing it

```powershell
cd backend
python scripts\prepare_dataset.py verify --root D:\datasets\LEVIR-CD          # structure + label sanity report
python scripts\prepare_dataset.py tile   --root D:\datasets\LEVIR-CD --out D:\datasets\LEVIR-CD-256
```

Tiling to 256×256 gives 7 120 / 1 024 / 2 048 tiles. It speeds up training because 1024×1024 images do not need
to be decoded again for every random crop. `train.py` also works on the original 1024 images, taking random 256 crops
(`--crops-per-image`).

Keep datasets **outside** the repository, or under `data/datasets/`, which is git-ignored.

## Bundled synthetic samples (local testing)

`data/samples/` contains three **procedurally generated** pairs: urban expansion, flood extent and vegetation
clearing. They are 512×512, with exact ground-truth masks, and were made by `backend/app/utils/synthetic.py`
(`python scripts/make_samples.py`). They are **not real satellite images** and are labelled synthetic everywhere
they appear. Each AFTER image includes a deliberate 1.2° rotation, a (5, −3) px shift, an illumination/colour change
and Gaussian noise, so that alignment, normalisation and evaluation are all exercised.

`prepare_dataset.py synthetic` writes a small LEVIR-style dataset of similar scenes. Use it to smoke-test the
training and evaluation code offline. **Never** present a model trained on it as meaningful.
