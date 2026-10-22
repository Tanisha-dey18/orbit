import { AlertTriangle, BookOpen, Brain, Cpu, Database, Workflow } from "lucide-react";
import { ModelStatusCard } from "../components/ResultPanels";
import { Notice, Panel, Spinner } from "../components/ui";
import type { BackendState } from "../hooks/useBackend";

const STEPS = [
  ["Ingest & validate", "Magic-byte format sniffing (PNG/JPEG/TIFF), size and pixel limits, corruption checks, alpha/nodata → validity mask, 16-bit percentile stretch."],
  ["Georeference", "If both images are GeoTIFFs and Rasterio is installed, AFTER is reprojected onto BEFORE's CRS and pixel grid."],
  ["Resize & cap", "Mismatched sizes are resized (with a warning - this is not geographic alignment). Very large images are downscaled for processing."],
  ["Register", "ORB keypoints + ratio test + RANSAC similarity transform; ECC fallback. Implausible transforms, or ones that lower gradient correlation, are rejected and the comparison is flagged."],
  ["Normalise", "AFTER's colour distribution is matched to BEFORE (mean/std or histogram) over valid pixels to suppress lighting differences."],
  ["Detect", "Classical: CVA or grayscale difference → intensity in [0,1]. Deep: FC-Siam-diff → per-pixel change probability, via tiled inference."],
  ["Threshold & clean", "Manual or Otsu threshold → morphological opening/closing → removal of components smaller than the minimum region size."],
  ["Quantify & interpret", "Changed-pixel percentage over valid pixels, connected regions, spatial grid, area (only with valid CRS metadata), rule-based heuristic categories."],
];

export default function MethodologyPage({ backend }: { backend: BackendState }) {
  const deep = backend.kind === "online" ? backend.models.methods.find((m) => m.id === "deep") : undefined;
  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-6">
      <div>
        <p className="label">Methodology</p>
        <h1 className="font-display text-3xl font-semibold">How PROJECT ORBIT works</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
          Every number in the dashboard comes from the pipeline below, computed on the images you provide. Nothing is
          simulated. When a step cannot be performed reliably, the result says so.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <Panel title="Pipeline" icon={<Workflow className="h-3.5 w-3.5" />}>
          <ol className="space-y-3">
            {STEPS.map(([t, d], i) => (
              <li key={t} className="flex gap-3">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-cyan/40 font-mono text-[11px] text-cyan">{i + 1}</span>
                <div><p className="text-sm font-medium text-ink">{t}</p><p className="text-xs leading-relaxed text-muted">{d}</p></div>
              </li>
            ))}
          </ol>
        </Panel>
        <div className="space-y-4">
          {backend.kind === "loading" && <Panel><Spinner /></Panel>}
          {backend.kind === "offline" && <Notice tone="error" title="Backend offline">{backend.error}</Notice>}
          <ModelStatusCard model={deep?.model} />
          <Panel title="Two methods, clearly labelled" icon={<Cpu className="h-3.5 w-3.5" />}>
            <p className="text-sm leading-relaxed text-muted">
              <span className="text-cyan">Classical baseline</span> results are deterministic image processing - they react to any
              radiometric difference. <span className="text-violet">Deep model</span> results come only from a loaded checkpoint;
              if none is available the API refuses deep requests instead of silently substituting the baseline.
            </p>
          </Panel>
          <Panel title="Training data" icon={<Database className="h-3.5 w-3.5" />}>
            <p className="text-sm leading-relaxed text-muted">
              Primary dataset: <span className="text-ink">LEVIR-CD</span> (637 pairs of 1024×1024 Google Earth images at 0.5 m/px, binary
              building-change labels; train/val/test 445/64/128). Research use only - check the dataset's terms before redistribution.
              See <code className="text-cyan">docs/dataset.md</code>.
            </p>
          </Panel>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Panel title="Deep-learning architecture" icon={<Brain className="h-3.5 w-3.5" />}>
          <p className="text-sm leading-relaxed text-muted">
            FC-Siam-diff (Daudt et al., ICIP 2018): a U-Net encoder with shared weights processes both images; at each scale the
            decoder receives |f<sub>before</sub> − f<sub>after</sub>|, focusing it on where features disagree. Output: one logit per
            pixel → sigmoid probability. ≈1.36 M parameters - trainable and runnable on CPU.
          </p>
        </Panel>
        <Panel title="Limitations" icon={<AlertTriangle className="h-3.5 w-3.5" />}>
          <ul className="list-disc space-y-1 pl-4 text-sm text-muted">
            <li>Changed-pixel % is a share of valid compared pixels, not of land area.</li>
            <li>Heuristic categories use RGB only; muddy water, shadows and bare soil are easily confused.</li>
            <li>A LEVIR-CD model detects building change; it is not validated for floods or vegetation.</li>
            <li>Imagery alone may not establish the cause or severity of a change.</li>
          </ul>
        </Panel>
      </div>
      <p className="flex items-center gap-2 text-xs text-dim"><BookOpen className="h-3.5 w-3.5" /> Full documentation: README.md, docs/architecture.md, docs/model_card.md, docs/limitations.md.</p>
    </div>
  );
}
