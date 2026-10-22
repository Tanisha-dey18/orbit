/**
 * Client-side threshold preview.
 *
 * The backend stores the change intensity (classical) or probability (deep model) as an
 * 8-bit PNG. Thresholding it in the browser lets the overlay update instantly while the
 * slider moves, without re-running the analysis. The preview applies the raw threshold
 * only; morphology + min-region filtering are applied server-side when "Apply" is pressed.
 */

export function loadImageData(url: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      if (!ctx) return reject(new Error("Canvas 2D context unavailable"));
      ctx.drawImage(img, 0, 0);
      resolve(ctx.getImageData(0, 0, c.width, c.height));
    };
    img.onerror = () => reject(new Error(`Failed to load ${url}`));
    img.src = url;
  });
}

/** Threshold an 8-bit intensity image -> RGBA overlay (change pixels coloured, others transparent). */
export function thresholdToOverlay(
  intensity: ImageData,
  threshold: number,
  color: [number, number, number] = [255, 64, 96],
): { overlay: ImageData; changed: number } {
  const t = Math.round(threshold * 255);
  const src = intensity.data;
  const out = new ImageData(intensity.width, intensity.height);
  const dst = out.data;
  let changed = 0;
  for (let i = 0; i < src.length; i += 4) {
    if (src[i] >= t && src[i] > 0) {
      dst[i] = color[0];
      dst[i + 1] = color[1];
      dst[i + 2] = color[2];
      dst[i + 3] = 255;
      changed++;
    }
  }
  return { overlay: out, changed };
}

export function countValid(valid: ImageData): number {
  let n = 0;
  for (let i = 0; i < valid.data.length; i += 4) if (valid.data[i] > 0) n++;
  return n;
}
