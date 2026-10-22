# Limitations

## Scientific / interpretive

1. **Change ≠ cause.** A changed pixel means the two images differ there. It does not mean a flood, collapse or
   disaster happened. Satellite imagery alone may not be enough to establish the cause or severity of a change.
2. **Heuristic categories.** "Building-related", "infrastructure", "vegetation" and "potential flood-related" come
   from simple RGB rules: excess-green index, darkness or blueness, low texture, low saturation, region shape. Known
   confusions: shadows and dark roofs look like water; muddy floodwater looks like bare soil; seasonal grass
   browning looks like vegetation loss; new cars and parking lots look like construction. Anything unclear is
   reported as *unclassified*.
3. **Changed-pixel percentage** is the share of *valid compared pixels* in the analysed image area. It is not a
   share of land area. It also depends on the threshold, the morphology settings and the image extent.
4. **Physical area** is reported only with valid CRS metadata (GeoTIFF + Rasterio). For geographic CRSs it is a
   spherical approximation. Downscaling for processing reduces the precision of the area.

## Technical

5. **Registration model.** A global similarity transform (shift, rotation, uniform scale) cannot correct terrain
   relief, building lean in off-nadir imagery, rolling-shutter or non-rigid distortion. These show up as false change
   along tall structures and slopes.
6. **Failed alignment** is detected heuristically: implausible transforms, too few inliers, gradient correlation below
   0.15, or less than 50% overlap. The checks are not perfect. A scene with very little texture can pass while still
   being slightly misaligned.
7. **Radiometric normalisation** is global. Local effects (cloud shadow, haze, different sun angle and therefore
   different shadow directions) remain and the classical baseline will flag them.
8. **Classical baseline** responds to any radiometric difference. It is useful as a transparent reference, not as
   a semantic detector.
9. **Deep model.** No trained weights are included. Once trained on LEVIR-CD it detects building change at ~0.5 m
   GSD, and it is not validated for other sensors, resolutions or change types.
10. **Bands.** For images with more than three bands, bands 1–3 are assumed to be R, G, B. NIR/SWIR are not used yet.
11. **Image size.** Images are downscaled to `ORBIT_MAX_PROCESSING_SIDE` (default 2048 px), so small changes can
    disappear in very large scenes. Tile the scene yourself for full resolution.
12. **TIFF previews.** Browsers cannot display TIFFs, so the dashboard shows a placeholder before analysis. The
    processed PNGs are shown afterwards.

## Operational

13. **CPU-only hardware.** Classical analysis is fast (~1 s for 512²). Deep inference on CPU scales with image area:
    expect several seconds per megapixel. Training on CPU is only practical for smoke tests; use a GPU (local, Colab
    or Kaggle) for real training.
14. **Single-node storage.** SQLite and local files are fine for one user or instance. Use Postgres and object storage
    for multi-instance deployments. Free hosting tiers often have ephemeral disks.
15. **No authentication.** Anyone who can reach the API can see and delete analyses. Do not expose it publicly with
    sensitive imagery.
16. **Synchronous requests.** Very large analyses hold an HTTP request open. A job queue is future work.

## Ethical

17. Imagery of populated areas can be sensitive. Follow the licences of your imagery providers, and do not use the
    system to make claims about specific people or properties without verification.
