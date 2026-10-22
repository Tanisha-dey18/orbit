# Model card: ORBIT FC-Siam-diff

## Status

| | |
|---|---|
| Trained checkpoint in repository | **None** |
| Training on a real dataset performed | **No** |
| Evaluated performance on real imagery | **Not established** |
| Code paths verified | architecture, training loop, checkpoint save/load, tiled inference, evaluation: by unit tests and a 3-epoch synthetic smoke run |

Until you train and evaluate the model yourself, the deep-learning mode of the app stays **disabled**, and the API
returns HTTP 409 for deep requests.

## Model details

* **Architecture:** FC-Siam-diff, a fully convolutional Siamese U-Net (Daudt, Le Saux & Boulch, "Fully Convolutional
  Siamese Networks for Change Detection", ICIP 2018). Independent re-implementation in `backend/app/models/siamese_unet.py`.
* **Encoder:** 4 levels (16/32/64/128 channels by default), Conv-BN-ReLU blocks with dropout, **shared weights** for
  BEFORE and AFTER.
* **Skip connections:** `|f_before − f_after|` at every level, so the decoder is driven by feature *differences*.
* **Decoder:** transposed-conv upsampling + Conv-BN-ReLU blocks → 1×1 conv → one logit per pixel.
* **Parameters:** ≈1.36 M (`base_channels=16, depth=4`).
* **Input:** two aligned RGB images, scaled to [0, 1], normalised with mean 0.5 and std 0.25 per channel (stored
  in the checkpoint as `metadata.normalization`).
* **Output:** per-pixel change probability (sigmoid). Binary mask = probability ≥ threshold (default 0.5, configurable).
* **Inference:** 256×256 tiles with 32 px overlap, averaged in overlaps; reflect/edge padding so any image size works.
  CPU and CUDA are both supported (`ORBIT_DEVICE`).

## Checkpoint format and verification

`torch.save` dict with `format="orbit-cd-v1"`, `arch`, `arch_config`, `state_dict` and `metadata` (model name and
version, dataset, normalisation, patch size, epochs, best validation metrics *as computed by `train.py`*, training
args, creation time). The loader:
1. uses `torch.load(weights_only=True)`, so no arbitrary code can run;
2. rejects files without the expected format tag or keys;
3. rebuilds the architecture from `arch_config` and calls `load_state_dict(strict=True)`, so any mismatch is an error;
4. records the SHA-256 of the file, which appears in every analysis result and report.

## Intended use

* Research, teaching and portfolio demonstration of bi-temporal change detection.
* Detecting **building-related** change in very-high-resolution (~0.5 m) RGB imagery similar to the training data.

## Out-of-scope use

* Determining the cause or severity of a change, or confirming a disaster.
* Flood, vegetation or damage mapping (not in LEVIR-CD labels).
* Medium-resolution imagery (e.g. Sentinel-2 at 10 m) without retraining.
* Any operational or safety-critical decision without expert review.

## Training data and procedure (when you train it)

* **Data:** LEVIR-CD official split (see `docs/dataset.md`). Optionally tiled to 256×256.
* **Augmentation:** random crop, horizontal and vertical flips, 90° rotations, random swap of temporal order.
* **Loss:** `BCEWithLogits(pos_weight=2) + Dice`.
* **Optimiser:** AdamW (lr 1e-3, weight decay 1e-4), cosine schedule.
* **Selection:** the epoch with the best validation F1 at threshold 0.5 is saved to `--out`; the last epoch goes to `*.last.pt`.
* **Command:** see README → *Dataset acquisition & training*.

## Evaluation

`scripts/evaluate.py` reports precision, recall, F1, IoU, accuracy and the confusion counts (summed over the split),
plus mean per-image F1, mean inference time, threshold and number of samples. **Fill in this table only with numbers
you have actually produced:**

| Split | Threshold | Precision | Recall | F1 | IoU | Samples | Checkpoint SHA-256 |
|---|---|---|---|---|---|---|---|
| LEVIR-CD test | 0.5 | *not yet evaluated* | | | | | |

## Limitations and risks

* Will learn LEVIR-CD's biases: Texas suburbs, Google Earth mosaics, mostly residential construction.
* Sensitive to misregistration, seasonal change, different sensors and off-nadir viewing.
* Probabilities are not calibrated. A threshold of 0.5 is a convention, not a guarantee.
* Results on a synthetic smoke-test checkpoint are meaningless. The app warns about them explicitly.

## License

The code is MIT-licensed (same as this repository). **Trained weights inherit the terms of the training data**: for
LEVIR-CD, academic research use.
