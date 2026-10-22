import { AlertTriangle, BarChart3, Brain, Download, FileJson, FileText, Grid3x3, Image as ImageIcon, Layers, Scale, Target } from "lucide-react";
import { useRef, useState } from "react";
import { Bar, BarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, artifactUrl, reportUrl } from "../services/api";
import type { AnalysisDetail, AnalysisResult, EvaluationResult, ModelInfo } from "../types/api";
import { fmtArea, fmtInt, fmtMetric, fmtMs, fmtPct } from "../utils/format";
import { Button, Notice, Panel, Spinner, Stat } from "./ui";

export function StatsPanel({ result, preview }: { result: AnalysisResult; preview?: { pct: number; threshold: number } | null }) {
  const s = result.statistics;
  const area = fmtArea(result.area.changed_area_m2);
  return (
    <Panel title="Change statistics" icon={<BarChart3 className="h-3.5 w-3.5" />}>
      <div className="grid grid-cols-2 gap-2">
        <div className="col-span-2">
          <Stat
            accent
            label="Changed pixels (of valid)"
            value={fmtPct(s.changed_pixel_percentage, 3)}
            hint="Share of valid compared pixels - not automatically the share of land area."
          />
        </div>
        <Stat label="Changed px" value={fmtInt(s.changed_pixel_count)} />
        <Stat label="Valid px" value={fmtInt(s.valid_pixel_count)} />
        <Stat label="Regions" value={fmtInt(s.region_count)} hint={s.largest_region_px ? `largest ${fmtInt(s.largest_region_px)} px` : undefined} />
        <Stat label="Processing" value={fmtMs(result.timing.total_ms)} hint={`detector ${fmtMs(result.timing.inference_ms)}`} />
        <div className="col-span-2">
          <Stat
            label="Approx. physical area"
            value={area ?? <span className="text-base text-dim">not estimated</span>}
            hint={result.area.explanation}
          />
        </div>
      </div>
      {preview && Math.abs(preview.threshold - result.threshold.value) > 1e-6 && (
        <p className="mt-2 font-mono text-[11px] text-warn">
          Preview @ {preview.threshold.toFixed(3)}: ≈{fmtPct(preview.pct, 2)} (raw threshold, before morphology) - press Apply to recompute.
        </p>
      )}
    </Panel>
  );
}

export function WarningsPanel({ result }: { result: AnalysisResult }) {
  const al = result.alignment;
  return (
    <Panel title="Reliability & warnings" icon={<AlertTriangle className="h-3.5 w-3.5" />}>
      <dl className="mb-3 grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[11px]">
        <dt className="text-dim">alignment</dt>
        <dd className={al.status === "failed" ? "text-change" : "text-ink"}>{String(al.method_used)} · {al.status}</dd>
        {Array.isArray(al.translation_px) && (<>
          <dt className="text-dim">shift / rot</dt>
          <dd>{(al.translation_px as number[]).join(", ")} px · {String(al.rotation_deg)}°</dd>
        </>)}
        <dt className="text-dim">struct. similarity</dt>
        <dd>{String(al.score_before)} → {String(al.score_after ?? al.score_before)}</dd>
        <dt className="text-dim">valid area</dt>
        <dd>{(al.valid_fraction * 100).toFixed(1)}%</dd>
        <dt className="text-dim">normalisation</dt>
        <dd>{String(al.normalization)}</dd>
      </dl>
      {result.warnings.length === 0 ? (
        <Notice tone="ok" title="No warnings raised" />
      ) : (
        <ul className="space-y-1.5">
          {result.warnings.map((w, i) => (
            <li key={i} className="rounded-md border border-warn/25 bg-warn/5 px-2.5 py-1.5 text-xs leading-snug text-warn/90">{w}</li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export function ModelStatusCard({ model, compact = false }: { model: ModelInfo | undefined; compact?: boolean }) {
  if (!model) return null;
  const meta = (model.metadata ?? {}) as Record<string, unknown>;
  return (
    <Panel title="Deep-learning model" icon={<Brain className="h-3.5 w-3.5" />}>
      <p className={`font-mono text-xs ${model.ready ? "text-violet" : "text-dim"}`}>
        status: {model.status} · device: {model.device}
      </p>
      <p className="mt-1.5 text-xs leading-relaxed text-muted">{model.message}</p>
      {model.ready && !compact && (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[11px]">
          {["model_name", "model_version", "dataset", "epochs_trained", "parameters"].map((k) => (
            <div key={k} className="contents">
              <dt className="text-dim">{k}</dt>
              <dd className="truncate text-ink">{String(meta[k] ?? "–")}</dd>
            </div>
          ))}
        </dl>
      )}
    </Panel>
  );
}

const CAT_COLORS: Record<string, string> = {
  infrastructure: "#3b82f6", building: "#22d3ee", vegetation: "#34d399", flood: "#a78bfa", unclassified: "#8a9bb4",
};

export function InterpretationPanel({ result, onSelectRegion, selectedRegion }: {
  result: AnalysisResult; onSelectRegion: (id: number | null) => void; selectedRegion: number | null;
}) {
  const it = result.interpretation;
  const cats = Object.entries(it.categories);
  return (
    <Panel title="Disaster intelligence (heuristic)" icon={<Layers className="h-3.5 w-3.5" />}>
      <ul className="mb-3 space-y-1.5 text-[13px] leading-snug text-ink/90">
        {it.summary.map((s, i) => <li key={i} className="flex gap-2"><span className="text-cyan">›</span><span>{s}</span></li>)}
      </ul>
      <table className="w-full text-xs">
        <thead><tr className="label text-left"><th className="pb-1 font-normal">Category</th><th className="pb-1 text-right font-normal">Regions</th><th className="pb-1 text-right font-normal">Share</th></tr></thead>
        <tbody>
          {cats.map(([k, c]) => (
            <tr key={k} className="border-t border-line/60">
              <td className="py-1 pr-2"><span className="mr-1.5 inline-block h-2 w-2 rounded-sm" style={{ background: CAT_COLORS[k] }} />{c.label}</td>
              <td className="py-1 text-right font-mono">{c.regions}</td>
              <td className="py-1 text-right font-mono">{(c.share_of_change * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
      {it.regions.length > 0 && (
        <div className="mt-3">
          <p className="label mb-1.5 flex items-center gap-1"><Target className="h-3 w-3" /> Regions - click to highlight</p>
          <div className="max-h-56 space-y-1 overflow-y-auto pr-1">
            {it.regions.slice(0, 30).map((r) => (
              <button
                key={r.id}
                onClick={() => onSelectRegion(selectedRegion === r.id ? null : r.id)}
                className={`block w-full rounded-md border px-2 py-1.5 text-left text-[11px] transition-colors ${
                  selectedRegion === r.id ? "border-cyan bg-cyan/10" : "border-line hover:border-cyan/40"
                }`}
              >
                <span className="flex items-center justify-between font-mono">
                  <span><span className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ background: CAT_COLORS[r.category] }} />#{r.id} {r.category}</span>
                  <span className="text-dim">{fmtInt(r.area_px)} px</span>
                </span>
                {selectedRegion === r.id && <span className="mt-1 block text-muted">{r.evidence.join("; ")}</span>}
              </button>
            ))}
          </div>
        </div>
      )}
      <p className="mt-3 text-[11px] leading-snug text-dim">{it.notice}</p>
    </Panel>
  );
}

export function HistogramPanel({ result, threshold }: { result: AnalysisResult; threshold: number }) {
  const data = result.histogram.map((h) => ({ x: h.bin_start + 1 / 64, count: h.count }));
  return (
    <Panel title={result.method.is_trained_model ? "Probability distribution" : "Difference-intensity distribution"} icon={<BarChart3 className="h-3.5 w-3.5" />}>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
            <XAxis dataKey="x" type="number" domain={[0, 1]} tickCount={6} tick={{ fill: "#5c6e88", fontSize: 10 }} stroke="#1c2c45" />
            <YAxis scale="sqrt" tick={{ fill: "#5c6e88", fontSize: 10 }} stroke="#1c2c45" />
            <Tooltip
              contentStyle={{ background: "#060d1a", border: "1px solid #1c2c45", fontSize: 11 }}
              labelFormatter={(v) => `intensity ≈ ${Number(v).toFixed(2)}`}
              formatter={(v) => [fmtInt(Number(v)), "pixels"]}
            />
            <Bar dataKey="count" fill="#3b82f6" isAnimationActive={false} />
            <ReferenceLine x={threshold} stroke="#ff4060" strokeDasharray="3 3" label={{ value: "threshold", fill: "#ff4060", fontSize: 10, position: "insideTopRight" }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-[11px] text-dim">Valid pixels only; y-axis is square-root scaled.</p>
    </Panel>
  );
}

export function SpatialGridPanel({ result }: { result: AnalysisResult }) {
  const g = result.statistics.spatial_grid_pct;
  const max = Math.max(0.0001, ...g.flat());
  return (
    <Panel title="Spatial distribution" icon={<Grid3x3 className="h-3.5 w-3.5" />}>
      <div className="grid grid-cols-3 gap-1">
        {g.flat().map((v, i) => (
          <div key={i} className="flex h-12 items-center justify-center rounded border border-line font-mono text-[11px]"
            style={{ background: `rgb(255 64 96 / ${0.08 + 0.6 * (v / max)})` }}>
            {v.toFixed(1)}%
          </div>
        ))}
      </div>
      <p className="mt-1.5 text-[11px] text-dim">% of valid pixels changed in each third of the scene (north at top, assuming north-up imagery). Change is {result.interpretation.spatial_distribution}.</p>
    </Panel>
  );
}

export function DownloadsPanel({ detail, version }: { detail: AnalysisDetail; version: string }) {
  const id = detail.summary.id;
  const items = [
    { href: reportUrl(id), label: "PDF report", icon: <FileText className="h-3.5 w-3.5" /> },
    { href: artifactUrl(id, "mask", { download: true, v: version }), label: "Change mask PNG", icon: <ImageIcon className="h-3.5 w-3.5" /> },
    { href: artifactUrl(id, "overlay", { download: true, v: version }), label: "Overlay PNG", icon: <Layers className="h-3.5 w-3.5" /> },
    { href: artifactUrl(id, "heatmap", { download: true }), label: "Heatmap PNG", icon: <BarChart3 className="h-3.5 w-3.5" /> },
    { href: artifactUrl(id, "results", { download: true, v: version }), label: "Results JSON", icon: <FileJson className="h-3.5 w-3.5" /> },
  ];
  return (
    <Panel title="Export" icon={<Download className="h-3.5 w-3.5" />}>
      <div className="grid grid-cols-1 gap-1.5">
        {items.map((it) => (
          <a key={it.label} href={it.href} download
            className="flex items-center gap-2 rounded-md border border-line bg-space-900/60 px-2.5 py-1.5 text-xs text-ink transition-colors hover:border-cyan/50 hover:text-cyan">
            {it.icon} {it.label}
          </a>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-dim">The report is generated from the stored results and reflects the current threshold.</p>
    </Panel>
  );
}

export function EvaluationPanel({ detail, sampleGroundTruth, onEvaluated }: {
  detail: AnalysisDetail; sampleGroundTruth?: (() => Promise<Blob>) | null; onEvaluated: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [res, setRes] = useState<EvaluationResult | null>(detail.result.evaluation);

  const run = async (blob: Blob, name?: string) => {
    setBusy(true);
    setError(null);
    try {
      setRes(await api.evaluate(detail.summary.id, blob, name));
      onEvaluated();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const m = res?.metrics;
  return (
    <Panel title="Evaluate vs ground truth" icon={<Scale className="h-3.5 w-3.5" />}>
      <p className="mb-2 text-xs text-muted">Upload a binary mask (white = change) in the BEFORE image frame to compute pixel-level metrics.</p>
      <div className="flex flex-wrap gap-2">
        <Button disabled={busy} onClick={() => input.current?.click()}>{busy ? <Spinner /> : null} Upload mask</Button>
        {sampleGroundTruth && (
          <Button disabled={busy} onClick={async () => {
            try { await run(await sampleGroundTruth(), "sample_ground_truth.png"); } catch (e) { setError(String(e)); }
          }}>Use sample ground truth</Button>
        )}
      </div>
      <input ref={input} type="file" accept=".png,.tif,.tiff" className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f, f.name); e.target.value = ""; }} />
      {error && <div className="mt-2"><Notice tone="error">{error}</Notice></div>}
      {m && (
        <div className="mt-3">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="F1" value={fmtMetric(m.f1)} accent />
            <Stat label="IoU" value={fmtMetric(m.iou)} />
            <Stat label="Precision" value={fmtMetric(m.precision)} />
            <Stat label="Recall" value={fmtMetric(m.recall)} />
          </div>
          <p className="mt-2 font-mono text-[11px] text-dim">
            TP {fmtInt(m.tp)} · FP {fmtInt(m.fp)} · FN {fmtInt(m.fn)} · TN {fmtInt(m.tn)} · threshold {res!.threshold.toFixed(3)} · {fmtInt(res!.evaluated_pixels)} px evaluated
          </p>
          {res!.warnings.map((w, i) => <p key={i} className="mt-1 text-[11px] text-warn">{w}</p>)}
        </div>
      )}
    </Panel>
  );
}
