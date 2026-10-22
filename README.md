# PROJECT ORBIT

**AI-powered satellite image change detection and disaster intelligence platform.**

PROJECT ORBIT compares two images of the same area taken at different times (BEFORE and AFTER).
It aligns them, finds the pixels that changed, measures the change, and explains it with cautious wording.
It has a real computer-vision pipeline: image validation, registration, a classical change-detection baseline,
and a Siamese U-Net deep-learning model with training and evaluation scripts. A FastAPI backend stores
analyses in SQLite, and a React dashboard displays the results.

> **Honesty first.** No result in this project is simulated. The classical baseline is always labelled as a baseline,
> never as AI. Deep-learning mode only works when a verified checkpoint is loaded. **No trained checkpoint is bundled**
> (see [Model status](#model-status)). A visual change is shown as evidence to investigate, never as proof of a disaster.

---

## Contents
1. [Problem statement](#problem-statement)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Technology stack](#technology-stack)
5. [Computer-vision pipeline](#computer-vision-pipeline)
6. [Model status](#model-status)
7. [Dataset acquisition & training](#dataset-acquisition--training)
8. [Evaluation methodology](#evaluation-methodology)
9. [Local installation (Windows PowerShell)](#local-installation-windows-powershell)
10. [Running tests](#running-tests)
11. [API documentation](#api-documentation)
12. [Example usage](#example-usage)
13. [Deployment](#deployment)
14. [Upload to GitHub](#upload-to-github)
15. [Known limitations](#known-limitations)
16. [Future improvements](#future-improvements)
17. [Further documentation](#further-documentation)

---

## Problem statement

After floods, earthquakes or rapid urban growth, analysts compare satellite images to answer one question: **what
changed, and where?** Doing this by eye is slow and inconsistent. Automating it is hard for four reasons:

* the two images are rarely pixel-aligned: they can differ in viewing angle, resolution, coverage and georeferencing;
* lighting, season, haze and sensor differences create change that is not real;
* a change mask alone does not tell you *why* something changed;
* overclaiming ("this building collapsed") is dangerous when the evidence is only a few pixels.

PROJECT ORBIT handles all four. It registers images explicitly and flags when registration fails. It normalises
radiometry. It offers both an interpretable baseline and a learned model. It reports interpretations as labelled
heuristics, together with the evidence behind them.

## Features

| Area | What is implemented |
|---|---|
| Ingestion | PNG/JPEG/TIFF, type detected from magic bytes (the extension is not trusted), size and pixel limits, corruption detection, alpha/nodata validity masks, 16-bit stretch, GeoTIFF CRS via Rasterio (optional) |
| Alignment | Geospatial reprojection, resize with warning, ORB + RANSAC similarity transform, ECC fallback, implausible-transform rejection, gradient-correlation quality check, valid-pixel tracking |
| Classical baseline | Change Vector Analysis, grayscale difference, Gaussian smoothing, manual or Otsu threshold, morphology, connected components |
| Deep learning | FC-Siam-diff (Siamese U-Net) in PyTorch, tiled CPU/GPU inference, verified checkpoint loading (`weights_only`, strict state dict), honest "unavailable" state |
| Training & evaluation | LEVIR-CD loader, augmentation, BCE + Dice loss, validation F1/IoU, best-checkpoint saving, evaluation script (precision/recall/F1/IoU/confusion/time) |
| Disaster intelligence | Rule-based RGB heuristics: infrastructure, building, vegetation, potential flood, **unclassified**; evidence per region; context never changes results |
| Area | Changed-pixel % over *valid* pixels; physical area only with a valid projected or geographic CRS |
| Dashboard | Upload or sample input, swipe slider, side-by-side, heatmap, mask, overlay with opacity, zoom/pan, live threshold preview, re-threshold without re-running the detector, region highlighting, histogram, spatial grid |
| History & export | SQLite history (paginated, filter, open, delete with files), PDF report, mask/overlay/heatmap PNG, JSON results, ground-truth evaluation |
| Engineering | FastAPI + Pydantic validation, structured logging, CORS from env, path-traversal protection, random filenames, concurrency limit, 59 backend tests + 9 frontend tests, Docker, Netlify/Vercel/Render configs |

## Architecture

```mermaid
flowchart LR
    subgraph Browser["React + TypeScript (Vite)"]
        UI[Landing / Analyze / History / Methodology]
        CV[Canvas threshold preview]
    end
    subgraph API["FastAPI backend"]
        R[REST routes<br/>api/analyses.py, api/system.py]
        P[pipeline.py orchestrator]
        IO[image_io.py<br/>validation]
        AL[alignment.py<br/>registration + normalisation]
        CL[classical.py<br/>CVA / grayscale diff]
        DL[deep.py<br/>FC-Siam-diff inference]
        MT[metrics.py + geo.py<br/>stats, area]
        IN[interpretation.py<br/>heuristics]
        RP[report.py<br/>PDF]
        ST[storage.py<br/>safe artifact store]
    end
    DB[(SQLite<br/>analyses, artifacts)]
    FS[[artifacts/analyses/&lt;id&gt;/*.png|pdf|json]]
    CK[[data/checkpoints/*.pt]]

    UI -- multipart upload / JSON --> R
    CV -- intensity.png --> R
    R --> P --> IO --> AL
    AL --> CL & DL
    CK -.-> DL
    CL & DL --> MT --> IN
    P --> ST --> FS
    P --> DB
    R --> RP --> ST
```

See [docs/architecture.md](docs/architecture.md) for the request lifecycle, the data model and the design decisions.

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | React 19, TypeScript, Vite 8, Tailwind CSS 4, Recharts, Lucide, React Router | typed, fast dev server, utility CSS keeps the custom dark theme consistent |
| API | FastAPI, Uvicorn, Pydantic v2, pydantic-settings | automatic validation and OpenAPI docs at `/docs`; async uploads with CPU work in a thread pool |
| Vision | OpenCV, NumPy, Pillow | mature registration (ORB, RANSAC, ECC), morphology, connected components |
| Geo | Rasterio *(optional)* | CRS-aware GeoTIFF reading and reprojection; skipped cleanly when not installed |
| Deep learning | PyTorch | FC-Siam-diff; ~1.36 M parameters, so it trains and runs on CPU |
| Storage | SQLAlchemy 2 + SQLite; files on local disk | zero-setup locally; swap the URL for Postgres in production |
| Reports | ReportLab | pure-Python PDF generation, nothing to install system-wide |

GeoPandas is **not** used: the pipeline never manipulates vector geometries. It would become relevant if changed
regions were exported as GeoJSON polygons (see future improvements).

## Computer-vision pipeline

```
upload ─► validate (magic bytes, size, pixels, corruption, channels)
       ─► [GeoTIFF pair] reproject AFTER onto BEFORE's CRS/grid
       ─► resize AFTER if sizes differ (warning: resizing ≠ georegistration)
       ─► downscale to ≤ ORBIT_MAX_PROCESSING_SIDE
       ─► register: ORB + ratio test + RANSAC similarity  →  ECC fallback
                    reject implausible transforms or ones that reduce gradient correlation
       ─► valid mask = BEFORE valid ∧ warped AFTER valid (borders eroded)
       ─► radiometric normalisation (mean/std or histogram matching)
       ─► change intensity  ─ classical: CVA ‖a−b‖ / (255√3)  or |gray(a)−gray(b)|/255
                            └ deep:      sigmoid(FC-Siam-diff(a, b))   (tiled, overlap-averaged)
       ─► threshold (manual or Otsu) ─► opening/closing ─► drop components < min_region_px
       ─► statistics, regions, 3×3 spatial grid, histogram, physical area (if CRS justifies it)
       ─► heuristic interpretation ─► artifacts (PNG/JSON) + SQLite record ─► PDF on demand
```

**Changed-pixel percentage** = `changed_pixel_count / valid_comparison_pixel_count × 100`. Pixels outside the
overlap after alignment, transparent pixels and nodata pixels are excluded. This is a share of the *analysed image
area*. It is not automatically the share of land area affected.

**Physical area** is reported only when the BEFORE image carries a CRS:
* projected CRS: `pixels × |a·e − b·d| × (metres per unit)²` (adjusted for any processing downscale);
* geographic CRS: per-row spherical cell area `R² · Δλ · |sin φ₂ − sin φ₁|` (authalic radius);
* anything else, including all PNG/JPEG files: **not estimated**. A pixel size is never invented.

## Model status

| Question | Answer (current state of this repository) |
|---|---|
| Is a trained checkpoint bundled? | **No.** `data/checkpoints/` is empty and the API reports `model.status = "missing"`. |
| Has training been performed on a real dataset? | **No.** LEVIR-CD cannot be downloaded automatically (manual download, research terms). |
| What *was* run? | A 3-epoch **smoke test** on a 30-pair *synthetic* dataset (CPU, ~75 s). It confirmed that training, checkpointing, loading and evaluation run end to end. That checkpoint is not shipped and says nothing about real-world accuracy. |
| What happens without weights? | The classical baseline keeps working. `method=deep` requests return **HTTP 409 `model_unavailable`**. Nothing is silently substituted. |
| Performance on real data | **Not yet established.** Train on LEVIR-CD and run `scripts/evaluate.py` to measure it. |

How to enable deep learning: follow [Dataset acquisition & training](#dataset-acquisition--training). Then place the
resulting `orbit_fc_siam_diff.pt` in `data/checkpoints/` (or set `ORBIT_MODEL_PATH`) and restart the backend.
Details are in [docs/model_card.md](docs/model_card.md).

## Dataset acquisition & training

Primary dataset: **LEVIR-CD**. It has 637 pairs of 1024×1024 RGB images at 0.5 m/pixel, binary building-change
labels, and an official 445/64/128 train/val/test split. Download it manually from the authors' page (linked in
[docs/dataset.md](docs/dataset.md)) and extract it so it looks like `LEVIR-CD/{train,val,test}/{A,B,label}/*.png`.

```powershell
cd backend
.\.venv\Scripts\Activate.ps1

# 1. verify the structure, pairing, sizes and label values
python scripts\prepare_dataset.py verify --root D:\datasets\LEVIR-CD

# 2. (recommended on CPU) cut into 256x256 tiles -> 7,120 / 1,024 / 2,048 tiles
python scripts\prepare_dataset.py tile --root D:\datasets\LEVIR-CD --out D:\datasets\LEVIR-CD-256

# 3. train (GPU strongly recommended for 100+ epochs; CPU works but takes many hours)
python scripts\train.py --data-root D:\datasets\LEVIR-CD-256 --dataset-name LEVIR-CD `
    --epochs 100 --batch-size 16 --lr 1e-3 --out ..\data\checkpoints\orbit_fc_siam_diff.pt

# 4. evaluate on the official test split (full 1024x1024 images)
python scripts\evaluate.py --data-root D:\datasets\LEVIR-CD --split test --method deep `
    --checkpoint ..\data\checkpoints\orbit_fc_siam_diff.pt --out ..\docs\results\levir_test.json
```

Smoke test without downloading anything (verifies the code only, **not** a usable model):

```powershell
python scripts\prepare_dataset.py synthetic --out ..\data\synthetic_cd
python scripts\train.py --data-root ..\data\synthetic_cd --dataset-name synthetic-smoke-test `
    --epochs 2 --out ..\data\checkpoints\smoke_test.pt
```

If you load a checkpoint whose dataset name starts with `synthetic`, every analysis that uses it carries a warning.

## Evaluation methodology

* Pixel-level binary metrics for the *change* class: precision, recall, F1, IoU, accuracy and TP/FP/FN/TN.
* Dataset metrics come from confusion counts summed over all test images (the usual LEVIR-CD convention). The mean
  per-image F1 is reported too.
* The threshold, the number of samples and the mean inference time are always recorded.
* A metric whose denominator is zero is reported as `null` ("undefined"), never as 0 or 1.
* In the app, `POST /api/evaluate` scores a stored mask against an uploaded ground-truth mask (BEFORE frame), using only
  valid pixels.

Measured so far (real runs, reproducible):

| What | Data | Threshold | F1 | IoU | File |
|---|---|---|---|---|---|
| Classical CVA baseline | 3 bundled **synthetic** samples (with misalignment) | 0.12 | 0.959 | 0.920 | `docs/results/classical_synthetic_samples.json` |

These synthetic numbers only show that the pipeline works correctly. **They are not a benchmark**: the synthetic
scenes are far easier than real imagery.

## Local installation (Windows PowerShell)

Prerequisites: **Python 3.11–3.13**, **Node.js 20+**, Git. A GPU is optional.

```powershell
# 0. get the code
git clone https://github.com/<you>/project-orbit.git
cd project-orbit
Copy-Item .env.example .env          # optional: defaults work without it

# 1. backend virtual environment
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1         # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip

# 2. backend dependencies (CPU-only PyTorch is the smallest download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-geo.txt  # optional: GeoTIFF CRS support (rasterio)

# 3. generate the bundled synthetic sample pairs (already committed; re-run any time)
python scripts\make_samples.py

# 4. start the API  ->  http://127.0.0.1:8000/docs
uvicorn app.main:app --reload --port 8000
```

In a **second** PowerShell window:

```powershell
cd project-orbit\frontend
npm install
npm run dev                          # -> http://localhost:5173
```

In development the Vite server proxies `/api` to `http://127.0.0.1:8000`, so CORS needs no configuration.
Open http://localhost:5173, click **Launch Analysis**, choose **Sample pair → Synthetic urban expansion**, then **Run analysis**.

Production frontend build: `npm run build` (output in `frontend/dist/`), preview with `npm run preview`.

## Running tests

```powershell
# backend (from backend/, venv active)
python -m pytest -q

# frontend (from frontend/)
npm run typecheck
npm test
npm run build
```

What the backend tests cover: image validation, corrupted and unsupported files, upload limits, different
dimensions, aspect-ratio mismatch, failed alignment, recovery of shift and rotation, radiometric normalisation,
classical detection, empty masks, exact percentage calculations, valid-pixel exclusion, morphology, Otsu, metric
edge cases, projected and geographic area, the context-independence of the heuristics, architecture forward pass,
checkpoint round-trip, mismatched and foreign checkpoints, missing weights, tiled inference at odd sizes, every API
endpoint, deep requests rejected without a model, deep inference with a loaded checkpoint, database persistence,
artifact retrieval, PDF generation, re-thresholding, evaluation, path traversal, error responses without leaked
paths, batch limits, and deletion of files.

## API documentation

Interactive OpenAPI docs: **http://127.0.0.1:8000/docs**.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | backend health **and** model readiness (separately) |
| GET | `/api/models` | available methods, model status/metadata, limits, contexts |
| GET | `/api/samples` · `/api/samples/{id}/{before,after,ground_truth}` | bundled synthetic samples |
| POST | `/api/analyze` | multipart: `before`, `after`, `params` (JSON string) → 201 + full result |
| POST | `/api/analyze/batch` | multipart: `befores[]`, `afters[]`, `params` (max `ORBIT_MAX_BATCH_PAIRS`) |
| GET | `/api/analyses?page=&page_size=&method=` | paginated history |
| GET | `/api/analyses/{id}` | summary + full result + artifact list |
| DELETE | `/api/analyses/{id}` | deletes DB rows **and** the analysis folder |
| POST | `/api/analyses/{id}/rethreshold` | new threshold/Otsu from the stored intensity map (no detector re-run) |
| GET | `/api/analyses/{id}/artifacts` | artifact metadata |
| GET | `/api/analyses/{id}/artifacts/{type}[?download=true]` | `before, after, intensity, heatmap, mask, overlay, valid, results, report` |
| GET | `/api/analyses/{id}/report` | generate (cached) and download the PDF |
| GET | `/api/analyses/{id}/mask` · `/overlay` | shortcuts for the PNGs |
| POST | `/api/evaluate` | multipart: `analysis_id`, `ground_truth` mask → precision/recall/F1/IoU |

`params` fields: `method` (`classical`/`deep`), `classical_method` (`cva`/`grayscale_diff`), `threshold` (0–1 or null),
`threshold_mode` (`manual`/`otsu`), `alignment` (`auto`/`feature`/`ecc`/`none`), `normalization`
(`mean_std`/`histogram`/`none`), `blur_sigma`, `morph_kernel`, `min_region_px`, `context`
(`general`/`urban`/`flood`/`vegetation`/`post_disaster`).

Errors always have the form `{"error": {"code": "...", "message": "..."}}`. Status codes: 404 unknown ID, 409 deep
model unavailable, 413 file too large, 422 invalid image/parameters or comparison not possible, 503 server busy.

## Example usage

```powershell
# classical analysis of a sample pair with curl.exe (ships with Windows 10+)
curl.exe -F "before=@data/samples/flood_extent_before.png" `
         -F "after=@data/samples/flood_extent_after.png" `
         -F "params={\"method\":\"classical\",\"threshold\":0.12,\"context\":\"flood\"}" `
         http://127.0.0.1:8000/api/analyze
```

```python
import json, requests
r = requests.post("http://127.0.0.1:8000/api/analyze",
                  files={"before": open("before.png", "rb"), "after": open("after.png", "rb")},
                  data={"params": json.dumps({"method": "classical", "threshold_mode": "otsu"})})
res = r.json()["result"]
print(res["statistics"]["changed_pixel_percentage"], res["alignment"]["status"], res["warnings"])
```

## Deployment

The frontend is static. The backend is a Python ML service that needs a long-running process, roughly 1 GB of RAM,
and a writable disk. **Netlify or Vercel cannot run the backend**, so deploy the two parts separately.

**Backend (Docker: Render, Railway, Fly.io, Hugging Face Spaces, any VPS)**
1. Build from the repo root: `docker build -f backend/Dockerfile -t orbit-api .`
2. Set `ORBIT_ALLOWED_ORIGINS=https://<your-frontend-domain>`.
3. **Persistence:** SQLite and the artifacts live in `/srv/artifacts`. Mount a persistent volume or disk there. Many
   free tiers have *ephemeral* filesystems, which means history disappears on every redeploy. For multi-instance
   production, use Postgres (`ORBIT_DATABASE_URL=postgresql+psycopg://…`, add `psycopg[binary]`) and object storage.
4. To enable deep mode, mount the checkpoint at `/srv/data/checkpoints/orbit_fc_siam_diff.pt`.
5. Render: `render.yaml` is a ready blueprint. Persistent disks need a paid instance type on Render.

**Frontend (Netlify / Vercel)**
1. Base directory `frontend/`, build command `npm run build`, publish directory `dist` (`netlify.toml` and `vercel.json` are included).
2. Set the environment variable `VITE_API_BASE_URL=https://<your-backend-domain>` (it is baked in at build time).
3. Make sure the backend's `ORBIT_ALLOWED_ORIGINS` contains the frontend URL exactly (scheme + host, no trailing slash).

**Everything with Docker Compose (local or a VPS):** `docker compose up --build` → http://localhost:8080 (nginx serves
the SPA and proxies `/api`).

CPU note: classical analysis of a 512×512 pair takes about 1 s on a laptop CPU. Deep inference for a 2048×2048 image
on CPU takes seconds to tens of seconds, depending on the processor. Lower `ORBIT_MAX_PROCESSING_SIDE` on small instances.

## Upload to GitHub

```powershell
cd project-orbit
git init
git add .
git status                      # check: no .env, .venv, node_modules, artifacts/*, *.pt
git commit -m "PROJECT ORBIT: satellite change detection platform"
git branch -M main
git remote add origin https://github.com/<you>/project-orbit.git
git push -u origin main
```

`.gitignore` already excludes secrets, virtual environments, `node_modules`, generated artifacts, datasets and
checkpoints. Publish trained weights as a **GitHub Release asset** or on the Hugging Face Hub, and check that the
dataset's terms allow you to share derived weights.

## Known limitations

Summary (full list in [docs/limitations.md](docs/limitations.md)):
* No trained checkpoint is included, so deep-learning performance is **not established**.
* A LEVIR-CD-trained model detects *building* change. It is not validated for floods, vegetation or damage.
* The heuristic categories use RGB only. Water, shadow, muddy floodwater and bare soil are easily confused.
* Registration is a global similarity transform. It cannot correct terrain relief, off-nadir parallax or non-rigid distortion.
* Without Rasterio, GeoTIFFs lose their CRS, so no physical area is reported.
* The bundled samples are synthetic and much easier than real imagery.

## Future improvements

* Train and publish a LEVIR-CD checkpoint with a full model card; add stronger architectures (BIT, ChangeFormer, SNUNet).
* Multispectral support (NIR/SWIR → NDVI/NDWI) to replace the RGB heuristics with physically meaningful indices.
* A validated semantic model (e.g. xView2/xBD damage classes) for building-damage assessment.
* Export changed regions as GeoJSON polygons (GeoPandas/Shapely) for use in QGIS.
* A task queue (Celery/RQ) for large scenes, Postgres and object storage for multi-user deployment, and authentication.

## Further documentation

* [docs/architecture.md](docs/architecture.md): components, request lifecycle, data model, design decisions
* [docs/dataset.md](docs/dataset.md): LEVIR-CD, WHU-CD, S2Looking; access, structure, splits, licensing
* [docs/model_card.md](docs/model_card.md): FC-Siam-diff model card
* [docs/limitations.md](docs/limitations.md): technical and ethical limitations
* [docs/viva.md](docs/viva.md): viva / interview preparation, likely questions with answers
* [data/README.md](data/README.md): what lives in `data/`
