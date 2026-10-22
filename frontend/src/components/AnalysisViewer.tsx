import { Columns2, Flame, Layers, ScanLine, SquareDashed } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { artifactUrl } from "../services/api";
import type { AnalysisDetail } from "../types/api";
import { countValid, loadImageData, thresholdToOverlay } from "../utils/threshold";
import ImageViewer from "./ImageViewer";
import SwipeCompare from "./SwipeCompare";
import { Spinner } from "./ui";

export type ViewMode = "swipe" | "side" | "heatmap" | "mask" | "overlay";

const MODES: { id: ViewMode; label: string; icon: React.ReactNode }[] = [
  { id: "overlay", label: "Overlay", icon: <Layers className="h-3.5 w-3.5" /> },
  { id: "swipe", label: "Swipe", icon: <ScanLine className="h-3.5 w-3.5" /> },
  { id: "side", label: "Side by side", icon: <Columns2 className="h-3.5 w-3.5" /> },
  { id: "heatmap", label: "Difference heatmap", icon: <Flame className="h-3.5 w-3.5" /> },
  { id: "mask", label: "Binary mask", icon: <SquareDashed className="h-3.5 w-3.5" /> },
];

function RegionBox({ detail, regionId, w, h }: { detail: AnalysisDetail; regionId: number | null; w: number; h: number }) {
  const r = detail.result.interpretation.regions.find((x) => x.id === regionId);
  if (!r) return null;
  const [x, y, bw, bh] = r.bbox;
  const pad = Math.max(2, Math.round(Math.max(w, h) / 200));
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <rect x={x - pad} y={y - pad} width={bw + 2 * pad} height={bh + 2 * pad} fill="none" stroke="#22d3ee"
        strokeWidth={Math.max(1.5, w / 300)} strokeDasharray={`${w / 80} ${w / 160}`} />
    </svg>
  );
}

export default function AnalysisViewer({ detail, version, liveThreshold, opacity, selectedRegion, onPreviewStats }: {
  detail: AnalysisDetail;
  version: string;
  liveThreshold: number;
  opacity: number;
  selectedRegion: number | null;
  onPreviewStats: (s: { pct: number; threshold: number } | null) => void;
}) {
  const [mode, setMode] = useState<ViewMode>("overlay");
  const id = detail.summary.id;
  const [w, h] = detail.result.processed_size;
  const url = (k: string) => artifactUrl(id, k, { v: version });

  const canvas = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<{ intensity: ImageData; mask: ImageData; valid: number } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setLoadError(null);
    Promise.all([loadImageData(url("intensity")), loadImageData(url("mask")), loadImageData(url("valid"))])
      .then(([intensity, mask, valid]) => !cancelled && setData({ intensity, mask, valid: countValid(valid) }))
      .catch((e) => !cancelled && setLoadError(String(e.message ?? e)));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, version]);

  const applied = detail.result.threshold.value;
  const previewing = Math.abs(liveThreshold - applied) > 1e-6;

  useEffect(() => {
    if (!data || !canvas.current) return;
    const { overlay, changed } = previewing
      ? thresholdToOverlay(data.intensity, liveThreshold)
      : thresholdToOverlay(data.mask, 0.5);
    const c = canvas.current;
    c.width = overlay.width;
    c.height = overlay.height;
    c.getContext("2d")!.putImageData(overlay, 0, 0);
    onPreviewStats(previewing ? { pct: (changed / Math.max(1, data.valid)) * 100, threshold: liveThreshold } : null);
  }, [data, liveThreshold, previewing, onPreviewStats, mode]);

  const label = (t: string) => <span className="absolute left-2 top-2 rounded bg-space-950/80 px-1.5 py-0.5 font-mono text-[10px]">{t}</span>;

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-1" role="tablist">
        {MODES.map((m) => (
          <button key={m.id} role="tab" aria-selected={mode === m.id} onClick={() => setMode(m.id)}
            className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
              mode === m.id ? "border-cyan/60 bg-cyan/10 text-cyan" : "border-line text-muted hover:text-ink"
            }`}>
            {m.icon}{m.label}
          </button>
        ))}
      </div>

      {mode === "side" ? (
        <div className="grid gap-3 md:grid-cols-2">
          {(["before", "after"] as const).map((k) => (
            <ImageViewer key={k} width={w} height={h} caption={k === "before" ? "BEFORE (processed)" : "AFTER (aligned + normalised)"}>
              <img src={url(k)} alt={k} className="absolute inset-0 h-full w-full" draggable={false} />
              <RegionBox detail={detail} regionId={selectedRegion} w={w} h={h} />
            </ImageViewer>
          ))}
        </div>
      ) : (
        <ImageViewer width={w} height={h} caption={
          mode === "overlay" ? (previewing
            ? `Live preview at threshold ${liveThreshold.toFixed(3)} (raw threshold, before morphology). Press “Apply threshold” to recompute statistics.`
            : `Change mask at applied threshold ${applied.toFixed(3)} over the aligned AFTER image. Scroll to zoom, drag to pan, double-click to reset.`)
            : mode === "heatmap" ? (detail.result.method.is_trained_model ? "Model change probability (blue = low, red = high)." : "Classical difference intensity (blue = low, red = high) - baseline, not a model prediction.")
            : mode === "mask" ? "Binary change mask (white = changed) after thresholding and morphology."
            : "Drag the handle to compare BEFORE and AFTER."
        }>
          {mode === "swipe" && <SwipeCompare before={url("before")} after={url("after")} />}
          {mode === "heatmap" && (<><img src={url("heatmap")} alt="Difference heatmap" className="absolute inset-0 h-full w-full" draggable={false} />{label("HEATMAP")}</>)}
          {mode === "mask" && (<><img src={url("mask")} alt="Change mask" className="pixelated absolute inset-0 h-full w-full" draggable={false} />{label("MASK")}</>)}
          {mode === "overlay" && (
            <>
              <img src={url("after")} alt="After" className="absolute inset-0 h-full w-full" draggable={false} />
              <canvas ref={canvas} className="pixelated absolute inset-0 h-full w-full" style={{ opacity }} />
              {!data && !loadError && <div className="absolute inset-0 flex items-center justify-center"><Spinner className="h-6 w-6 text-cyan" /></div>}
              {loadError && <div className="absolute inset-x-0 bottom-0 bg-change/80 p-2 text-xs">Could not load overlay data: {loadError}</div>}
              {label(previewing ? "PREVIEW" : "OVERLAY")}
            </>
          )}
          {mode !== "swipe" && <RegionBox detail={detail} regionId={selectedRegion} w={w} h={h} />}
        </ImageViewer>
      )}
    </div>
  );
}
