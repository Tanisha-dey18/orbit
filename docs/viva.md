# Viva / interview preparation

Likely questions about PROJECT ORBIT, with answers. Each answer points to the code.

---

### 1. What problem does the project solve, in one sentence?
Given two images of the same place at different times, it finds where the scene changed and measures it. It also
gives a careful, evidence-based interpretation, and it says clearly when the comparison is unreliable.

### 2. Walk me through the pipeline.
Validate → (reproject GeoTIFF) → resize if needed → downscale cap → register (ORB + RANSAC, ECC fallback) → valid
mask → radiometric normalisation → change intensity (classical or Siamese U-Net) → threshold → morphology →
connected components → statistics/area → heuristic interpretation → artifacts + DB. See `services/pipeline.py`.

### 3. Why is image registration necessary? What happens without it?
Change detection compares pixel (r, c) in BEFORE with pixel (r, c) in AFTER. A shift of 3 px makes every edge look
changed. On the bundled urban sample, gradient correlation is 0.26 before alignment and 0.87 after it. With the CVA
baseline at threshold 0.12, skipping alignment raises the changed area from 1.16% to 8.80% of pixels, and F1 against
ground truth falls from 0.72 to 0.13: the mask fills with false edges.

### 4. How does your registration work?
ORB detects corner-like keypoints with binary descriptors. I match AFTER→BEFORE with a Hamming brute-force matcher
and Lowe's ratio test (0.75). `estimateAffinePartial2D` with RANSAC then fits a 4-DOF similarity transform
(shift, rotation, uniform scale) and rejects outliers. If that fails, ECC maximises the correlation coefficient
directly on intensities (Euclidean motion), on a downsampled image for speed.

### 5. How do you know registration *failed*?
Several checks: too few keypoints, matches or inliers; an inlier ratio below 0.2; an implausible transform (scale
outside 0.8–1.25, rotation above 30°, shift above 30% of the image); and the warped result must not *lower* gradient
correlation. After registration, a low final correlation (< 0.15) or overlap below 50% marks the comparison
**unreliable**, and the UI shows a red banner.

### 6. Why a similarity transform and not a homography?
Near-nadir satellite images differ mostly by translation, rotation and scale. A homography has 8 DOF and can fit
degenerate perspective warps to noisy matches. Fewer degrees of freedom make RANSAC more robust. The trade-off is that
terrain relief and off-nadir parallax cannot be corrected (see limitations).

### 7. Why does resizing two images not make them aligned?
Resizing only matches pixel *counts*. If the images cover different ground extents, or are offset or rotated, the
same pixel index still points at different places on the ground. For GeoTIFFs we reproject with the CRS and
geotransform. For plain images we register by content.

### 8. Explain Change Vector Analysis.
Each pixel is an RGB vector. CVA takes the magnitude of the difference vector `‖a − b‖`, normalised by the maximum
`255·√3`. Unlike grayscale differencing, it catches changes in colour that keep brightness the same, for example
green vegetation becoming brown soil.

### 9. Why radiometric normalisation?
Different dates have different sun angles, atmosphere and sensor gain. Matching AFTER's per-channel mean and std (or
its full histogram) to BEFORE over valid pixels removes the global part of this difference, which reduces false change.

### 10. What does Otsu's method do?
It picks the threshold that maximises the between-class variance of the intensity histogram, which means it assumes
two populations (changed and unchanged). It works well when change is a substantial share of the image. When change
is very rare, the histogram is not bimodal and Otsu can choose poorly.

### 11. Why morphology and minimum region size?
Opening removes isolated speckle from noise. Closing fills small holes inside real changes. Dropping connected
components smaller than N pixels removes residual noise. Together they turn a noisy per-pixel threshold into
coherent regions.

### 12. Explain FC-Siam-diff.
It is a U-Net whose encoder is applied to both images with **shared weights** (Siamese), so both are embedded the same
way. The decoder's skip connections carry `|f_before − f_after|` at each scale instead of the raw features, so the
decoder learns from where the features disagree, at several resolutions. The output is one logit per pixel; sigmoid
turns it into a probability. Parameter count ≈ 1.36 M.

### 13. Why shared weights?
If the two branches had different weights, identical inputs could produce different features, and the "difference"
would be meaningless. Shared weights guarantee that an unchanged area gives near-zero feature differences, and they
halve the parameter count.

### 14. Why BCE + Dice loss?
Change pixels are rare (a few percent in LEVIR-CD). Plain BCE is dominated by the "no change" class. A positive-class
weight rebalances it, and Dice directly optimises overlap, which matches the F1/IoU metrics we report.

### 15. How do you run inference on images larger than the training patches?
Sliding 256×256 tiles with 32 px overlap; probabilities are averaged where tiles overlap. Edges are reflect-padded
(or edge-padded for tiny images) so dimensions are divisible by the network stride (16). See `ModelService.predict`.

### 16. Is your model trained? What accuracy does it have?
No trained checkpoint is included. The training pipeline is implemented and smoke-tested on synthetic data (3
epochs) to prove it runs. Real performance on LEVIR-CD has **not been established**, and I don't quote numbers I
haven't measured. The app disables deep mode until a verified checkpoint is loaded.

### 17. What happens if someone requests deep learning without weights?
The API returns HTTP 409 `model_unavailable`. It never silently runs the classical method and labels it as AI. Every
result also carries `method.is_trained_model`, and the UI shows a "Classical baseline · not AI" or "Deep model output"
badge.

### 18. How do you make sure a checkpoint matches the architecture?
`load_checkpoint` checks a format tag, rebuilds the network from the stored `arch_config`, and calls
`load_state_dict(strict=True)`, so missing or unexpected keys and shape mismatches raise errors. It uses
`weights_only=True`, so a malicious pickle cannot execute code. The file's SHA-256 is recorded in every result.

### 19. Define precision, recall, F1 and IoU for change detection.
For the change class: precision = TP/(TP+FP) (how many flagged pixels really changed); recall = TP/(TP+FN) (how much
real change we found); F1 = 2TP/(2TP+FP+FN), the harmonic mean; IoU = TP/(TP+FP+FN). We report `undefined` when a
denominator is zero, for example precision when nothing is predicted.

### 20. Why sum confusion counts over the dataset instead of averaging per image?
Per-image F1 is unstable for images with little or no change: one false positive gives F1 = 0. Summed counts weight
every pixel equally, which is the LEVIR-CD convention. I report both.

### 21. Why is changed-pixel percentage not the same as area affected?
It is relative to the *valid analysed pixels*. Those exclude no-overlap borders. The image extent is arbitrary. The
threshold and morphology change the value. Physical area also needs a known ground sampling distance, which PNG and
JPEG files do not carry.

### 22. How do you compute area for a geographic CRS (degrees)?
Degrees do not have a constant length. For each row, the cell area on a sphere is `R²·Δλ·|sin φ₂ − sin φ₁|` with the
authalic radius. Summing changed pixels × row cell area gives an approximation. For projected CRSs it is
`|det(affine)| × unit²` per pixel.

### 23. How does the disaster-intelligence module avoid overclaiming?
Every category is a named RGB heuristic with the evidence shown, for example "green-dominant pixels 78% → 4%".
Floods are always "*potential* flood-related". Anything that does not clearly meet a rule is "unclassified". The
selected context only changes which findings the summary emphasises, and a unit test proves it cannot change the
categories.

### 24. Why are the bundled samples synthetic? Isn't that cheating?
They exist so the whole system can be tested offline with exact ground truth, and they are labelled synthetic
everywhere. They include realistic nuisances (rotation, shift, lighting, noise). They are not used to claim real-world
accuracy; LEVIR-CD is the documented dataset for that.

### 25. How is the threshold slider instant without rerunning the model?
The backend stores the change intensity or probability as an 8-bit PNG. The browser thresholds it on a canvas while
you drag. "Apply" calls `/rethreshold`, which recomputes morphology, statistics and interpretation from the same PNG.
Intensities are quantised to 1/255 at analysis time, so re-applying the original threshold reproduces the original
mask exactly (this is tested).

### 26. What security measures did you implement?
Magic-byte type checks, byte and pixel limits (decompression bombs), random server-side IDs and file names, regex
validation of IDs, artifact access by *type* only, a check that resolved paths stay under the root, sanitised display
names, generic 500 errors, a CORS allow-list, and `weights_only` model loading. Missing for production: authentication
and rate limiting.

### 27. Why SQLite? Would it scale?
It needs no setup and fits a local tool. SQLAlchemy keeps the code database-agnostic: changing `ORBIT_DATABASE_URL` to
Postgres is enough for a multi-instance deployment, where object storage would also replace local files.

### 28. Why can't the backend be deployed on Netlify?
Netlify and Vercel host static files and short-lived serverless functions. This backend needs PyTorch (hundreds of MB),
long CPU-bound requests and a writable disk. It is deployed as a Docker service (Render, Railway, a VPS), and the
static frontend calls it through `VITE_API_BASE_URL`.

### 29. What would you improve next?
Train and evaluate on LEVIR-CD with a GPU; try attention-based models (BIT, ChangeFormer); use NIR/SWIR bands for
NDVI/NDWI-based categories; add a validated damage model (xBD); export regions as GeoJSON; add a job queue and authentication.

### 30. What was the hardest engineering problem?
Making results trustworthy rather than just producing a mask: detecting failed alignment, tracking which pixels are
valid, keeping baseline and model outputs strictly separate, and making re-thresholding reproducible.
