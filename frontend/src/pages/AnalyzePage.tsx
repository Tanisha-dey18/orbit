import { ChevronDown, FlaskConical, Play, Radar, RotateCcw, SlidersHorizontal, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import AnalysisViewer from "../components/AnalysisViewer";
import MethodBadge from "../components/MethodBadge";
import {
  DownloadsPanel, EvaluationPanel, HistogramPanel, InterpretationPanel, ModelStatusCard, SpatialGridPanel, StatsPanel,
  WarningsPanel,
} from "../components/ResultPanels";
import { Badge, Button, EmptyState, Notice, Panel, Spinner } from "../components/ui";
import UploadDropzone, { useLocalImage } from "../components/UploadDropzone";
import type { BackendState } from "../hooks/useBackend";
import { api, sampleUrl } from "../services/api";
import type { AnalysisDetail, AnalysisParams, Context, Sample } from "../types/api";
import { fmtDate, shortId } from "../utils/format";

const DEFAULT_PARAMS: AnalysisParams = {
  method: "classical", classical_method: "cva", threshold: 0.12, threshold_mode: "manual", alignment: "auto",
  normalization: "mean_std", blur_sigma: 1, morph_kernel: 3, min_region_px: 16, context: "general",
};
const CONTEXT_LABELS: Record<Context, string> = {
  general: "General", urban: "Urban growth", flood: "Flood", vegetation: "Vegetation", post_disaster: "Post-disaster",
};

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-1 block text-[11px] leading-snug text-dim">{hint}</span>}
    </label>
  );
}
const selectCls = "w-full rounded-md border border-line bg-space-900 px-2 py-1.5 text-sm text-ink focus:border-cyan focus:outline-none";

export default function AnalyzePage({ backend }: { backend: BackendState }) {
  const { id: routeId } = useParams();
  const navigate = useNavigate();
  const online = backend.kind === "online";
  const models = online ? backend.models : null;
  const deepInfo = models?.methods.find((m) => m.id === "deep");
  const maxMb = models?.limits.max_upload_mb ?? 25;

  // ---- inputs -----------------------------------------------------------------------------
  const [inputMode, setInputMode] = useState<"upload" | "sample">("upload");
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [samplesError, setSamplesError] = useState<string | null>(null);
  const [params, setParams] = useState<AnalysisParams>(DEFAULT_PARAMS);
  const [advanced, setAdvanced] = useState(false);
  const beforeImg = useLocalImage(beforeFile, !!sampleId);
  const afterImg = useLocalImage(afterFile, !!sampleId);

  // ---- results ----------------------------------------------------------------------------
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveThreshold, setLiveThreshold] = useState(0.12);
  const [opacity, setOpacity] = useState(0.65);
  const [selectedRegion, setSelectedRegion] = useState<number | null>(null);
  const [preview, setPreview] = useState<{ pct: number; threshold: number } | null>(null);
  const [applying, setApplying] = useState(false);

  const version = detail ? detail.result.rethresholded_at ?? detail.result.created_at : "";
  const onPreviewStats = useCallback((s: { pct: number; threshold: number } | null) => setPreview(s), []);

  const showDetail = (d: AnalysisDetail) => {
    setDetail(d);
    setLiveThreshold(d.result.threshold.value);
    setSelectedRegion(null);
  };

  useEffect(() => {
    if (!routeId) { setDetail(null); return; }
    if (detail?.summary.id === routeId) return;
    setLoadingDetail(true);
    setError(null);
    api.get(routeId).then(showDetail).catch((e) => setError(e.message)).finally(() => setLoadingDetail(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [routeId]);

  useEffect(() => {
    if (inputMode !== "sample" || samples || !online) return;
    api.samples().then(setSamples).catch((e) => setSamplesError(e.message));
  }, [inputMode, samples, online]);

  const pickSample = async (s: Sample) => {
    setError(null);
    try {
      const [b, a] = await Promise.all([
        api.fetchBlob(sampleUrl(s.id, "before"), `${s.id}_before.png`),
        api.fetchBlob(sampleUrl(s.id, "after"), `${s.id}_after.png`),
      ]);
      setSampleId(s.id);
      setBeforeFile(b);
      setAfterFile(a);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const setMethod = (method: AnalysisParams["method"]) => {
    const t = method === "deep" ? 0.5 : 0.12;
    setParams((p) => ({ ...p, method, threshold: t }));
  };

  const run = async () => {
    if (!beforeFile || !afterFile) return;
    setRunning(true);
    setError(null);
    try {
      const d = await api.analyze(beforeFile, afterFile, {
        ...params, threshold: params.threshold_mode === "otsu" ? null : params.threshold,
      });
      showDetail(d);
      navigate(`/analysis/${d.summary.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const applyThreshold = async (mode: "manual" | "otsu" = "manual") => {
    if (!detail) return;
    setApplying(true);
    setError(null);
    try {
      showDetail(await api.rethreshold(detail.summary.id, mode === "otsu" ? { threshold_mode: "otsu" } : { threshold: liveThreshold, threshold_mode: "manual" }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(false);
    }
  };

  const sampleGT = useMemo(() => {
    const sid = sampleId ?? (detail?.summary.before_filename.match(/^([a-z_]+)_before\.png$/)?.[1] ?? null);
    if (!sid || !online) return null;
    return () => api.fetchBlob(sampleUrl(sid, "ground_truth"), "gt.png");
  }, [sampleId, detail, online]);

  const canRun = online && !!beforeFile && !!afterFile && !running && (params.method === "classical" || !!deepInfo?.available);

  // ---- render -------------------------------------------------------------------------------
  return (
    <div className="mx-auto grid max-w-[1600px] gap-4 px-4 py-5 lg:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)_340px]">
      {/* LEFT: inputs ------------------------------------------------------------------------ */}
      <aside className="space-y-4 lg:row-span-2 xl:row-span-1">
        <Panel title="Input imagery" icon={<Upload className="h-3.5 w-3.5" />}>
          <div className="mb-3 grid grid-cols-2 gap-1 rounded-lg border border-line bg-space-900 p-0.5 text-xs">
            {(["upload", "sample"] as const).map((m) => (
              <button key={m} onClick={() => setInputMode(m)}
                className={`rounded-md py-1.5 transition-colors ${inputMode === m ? "bg-space-700 text-cyan" : "text-muted hover:text-ink"}`}>
                {m === "upload" ? "Upload pair" : "Sample pair"}
              </button>
            ))}
          </div>
          {inputMode === "sample" && (
            <div className="mb-3 space-y-1.5">
              {!online && <Notice tone="warn">Samples load from the backend, which is offline.</Notice>}
              {samplesError && <Notice tone="error">{samplesError}</Notice>}
              {online && !samples && !samplesError && <p className="flex items-center gap-2 text-xs text-muted"><Spinner /> loading samples…</p>}
              {samples?.length === 0 && <Notice tone="info">No samples found. Run <code>python scripts/make_samples.py</code>.</Notice>}
              {samples?.map((s) => (
                <button key={s.id} onClick={() => pickSample(s)}
                  className={`w-full rounded-md border px-2.5 py-2 text-left text-xs transition-colors ${sampleId === s.id ? "border-cyan bg-cyan/10" : "border-line hover:border-cyan/40"}`}>
                  <span className="flex items-center justify-between gap-2"><span className="font-medium text-ink">{s.title}</span>{s.synthetic && <Badge tone="violet">synthetic</Badge>}</span>
                  <span className="mt-0.5 block text-dim">{s.description}</span>
                </button>
              ))}
              <p className="text-[11px] leading-snug text-dim">Samples are procedurally generated test scenes, not real satellite images. They include ground truth for evaluation.</p>
            </div>
          )}
          <div className="space-y-3">
            <UploadDropzone label="Before (earlier)" image={beforeImg} maxMb={maxMb} onFile={(f) => { setBeforeFile(f); setSampleId(null); }} />
            <UploadDropzone label="After (later)" image={afterImg} maxMb={maxMb} onFile={(f) => { setAfterFile(f); setSampleId(null); }} />
          </div>
          {beforeImg?.width && afterImg?.width && (beforeImg.width !== afterImg.width || beforeImg.height !== afterImg.height) && (
            <div className="mt-3"><Notice tone="warn">Image sizes differ. AFTER will be resized and registered; resizing alone does not guarantee geographic alignment.</Notice></div>
          )}
        </Panel>

        <Panel title="Analysis settings" icon={<SlidersHorizontal className="h-3.5 w-3.5" />}>
          <div className="space-y-3">
            <Field label="Detection method">
              <div className="grid grid-cols-2 gap-1.5">
                <button onClick={() => setMethod("classical")}
                  className={`rounded-md border px-2 py-2 text-left text-xs ${params.method === "classical" ? "border-cyan bg-cyan/10 text-cyan" : "border-line text-muted"}`}>
                  Classical<span className="block text-[10px] text-dim">baseline, always available</span>
                </button>
                <button onClick={() => setMethod("deep")} disabled={!deepInfo?.available}
                  title={deepInfo?.available ? "" : deepInfo?.model?.message}
                  className={`rounded-md border px-2 py-2 text-left text-xs disabled:cursor-not-allowed disabled:opacity-50 ${params.method === "deep" ? "border-violet bg-violet/10 text-violet" : "border-line text-muted"}`}>
                  Deep learning<span className="block text-[10px] text-dim">{deepInfo?.available ? "FC-Siam-diff" : "unavailable - no weights"}</span>
                </button>
              </div>
            </Field>
            {params.method === "classical" && (
              <Field label="Classical algorithm">
                <select className={selectCls} value={params.classical_method}
                  onChange={(e) => setParams({ ...params, classical_method: e.target.value as AnalysisParams["classical_method"] })}>
                  <option value="cva">Change Vector Analysis (RGB)</option>
                  <option value="grayscale_diff">Absolute grayscale difference</option>
                </select>
              </Field>
            )}
            <Field label={`Sensitivity threshold · ${params.threshold_mode === "otsu" ? "auto" : (params.threshold ?? 0).toFixed(2)}`}
              hint={params.method === "deep" ? "Probability above which a pixel counts as changed." : "Lower = more sensitive (more change, more noise)."}>
              <input type="range" min={0.02} max={0.95} step={0.01} value={params.threshold ?? 0.12} disabled={params.threshold_mode === "otsu"}
                onChange={(e) => setParams({ ...params, threshold: Number(e.target.value) })} className="w-full disabled:opacity-40" />
              <label className="mt-1 flex items-center gap-2 text-xs text-muted">
                <input type="checkbox" checked={params.threshold_mode === "otsu"} onChange={(e) => setParams({ ...params, threshold_mode: e.target.checked ? "otsu" : "manual" })} />
                Automatic threshold (Otsu)
              </label>
            </Field>
            <Field label="Analysis context" hint="Only changes which findings the summary emphasises - never the detection result.">
              <select className={selectCls} value={params.context} onChange={(e) => setParams({ ...params, context: e.target.value as Context })}>
                {(models?.contexts ?? (Object.keys(CONTEXT_LABELS) as Context[])).map((c) => <option key={c} value={c}>{CONTEXT_LABELS[c]}</option>)}
              </select>
            </Field>
            <button onClick={() => setAdvanced(!advanced)} className="flex w-full items-center justify-between text-xs text-muted hover:text-cyan">
              Advanced preprocessing <ChevronDown className={`h-3.5 w-3.5 transition-transform ${advanced ? "rotate-180" : ""}`} />
            </button>
            {advanced && (
              <div className="space-y-3 border-l border-line pl-3">
                <Field label="Alignment">
                  <select className={selectCls} value={params.alignment} onChange={(e) => setParams({ ...params, alignment: e.target.value as AnalysisParams["alignment"] })}>
                    <option value="auto">Auto (ORB features → ECC fallback)</option>
                    <option value="feature">ORB features + RANSAC</option>
                    <option value="ecc">ECC intensity registration</option>
                    <option value="none">None (already co-registered)</option>
                  </select>
                </Field>
                <Field label="Radiometric normalisation">
                  <select className={selectCls} value={params.normalization} onChange={(e) => setParams({ ...params, normalization: e.target.value as AnalysisParams["normalization"] })}>
                    <option value="mean_std">Mean / std matching</option>
                    <option value="histogram">Histogram matching</option>
                    <option value="none">None</option>
                  </select>
                </Field>
                <div className="grid grid-cols-3 gap-2">
                  <Field label="Blur σ"><input type="number" min={0} max={5} step={0.5} value={params.blur_sigma} className={selectCls} onChange={(e) => setParams({ ...params, blur_sigma: Number(e.target.value) })} /></Field>
                  <Field label="Morph k"><input type="number" min={0} max={15} step={2} value={params.morph_kernel} className={selectCls} onChange={(e) => setParams({ ...params, morph_kernel: Number(e.target.value) })} /></Field>
                  <Field label="Min px"><input type="number" min={0} max={100000} step={4} value={params.min_region_px} className={selectCls} onChange={(e) => setParams({ ...params, min_region_px: Number(e.target.value) })} /></Field>
                </div>
              </div>
            )}
            <Button variant="primary" className="w-full !py-2.5" disabled={!canRun} onClick={run}>
              {running ? <><Spinner /> Analysing…</> : <><Play className="h-4 w-4" /> Run analysis</>}
            </Button>
            {!online && <p className="text-[11px] text-change">Backend offline - start it with <code>uvicorn app.main:app --port 8000</code>.</p>}
            {online && (!beforeFile || !afterFile) && <p className="text-[11px] text-dim">Select a BEFORE and an AFTER image to enable analysis.</p>}
          </div>
        </Panel>
      </aside>

      {/* MAIN: visualisation ----------------------------------------------------------------------- */}
      <section className="min-w-0 space-y-4 xl:col-start-2 xl:row-start-1">
        {error && <Notice tone="error" title="Something went wrong">{error}</Notice>}
        <Panel className="min-h-[420px]">
          {running || loadingDetail ? (
            <EmptyState icon={<Radar className="h-6 w-6 animate-pulse" />} title={running ? "Analysing image pair…" : "Loading analysis…"}>
              {running && "Validating → aligning → normalising → detecting change → computing statistics. Large images on CPU can take a while."}
            </EmptyState>
          ) : !detail ? (
            <EmptyState icon={<FlaskConical className="h-6 w-6" />} title="No analysis yet">
              Upload a BEFORE/AFTER pair or pick a sample, choose a method and press <span className="text-cyan">Run analysis</span>.
              Results appear here with swipe comparison, heatmap, mask and overlay views.
            </EmptyState>
          ) : (
            <>
              <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="label">Analysis {shortId(detail.summary.id)} · {fmtDate(detail.summary.created_at)}</p>
                  <p className="mt-1 truncate font-display text-lg">{detail.result.method.label}</p>
                  <p className="truncate text-xs text-dim">{detail.summary.before_filename} → {detail.summary.after_filename} · {detail.result.processed_size.join("×")} px processed</p>
                </div>
                <MethodBadge result={detail.result} />
              </div>
              {!detail.result.reliable && (
                <div className="mb-3"><Notice tone="error" title="Unreliable comparison">The images could not be compared reliably (see warnings). Detected changes may be artefacts of misalignment or differing coverage.</Notice></div>
              )}
              <AnalysisViewer detail={detail} version={version} liveThreshold={liveThreshold} opacity={opacity}
                selectedRegion={selectedRegion} onPreviewStats={onPreviewStats} />
              <div className="mt-4 grid gap-4 rounded-lg border border-line bg-space-900/50 p-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
                <Field label={`Threshold · ${liveThreshold.toFixed(3)} (applied ${detail.result.threshold.value.toFixed(3)}${detail.result.threshold.mode === "otsu" ? ", Otsu" : ""})`}>
                  <input type="range" min={0.01} max={0.99} step={0.005} value={liveThreshold} onChange={(e) => setLiveThreshold(Number(e.target.value))} className="w-full" />
                </Field>
                <Field label={`Overlay opacity · ${Math.round(opacity * 100)}%`}>
                  <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="w-full" />
                </Field>
                <div className="flex gap-2">
                  <Button onClick={() => applyThreshold("manual")} disabled={applying || Math.abs(liveThreshold - detail.result.threshold.value) < 1e-6}>
                    {applying ? <Spinner /> : null} Apply threshold
                  </Button>
                  <Button variant="ghost" title="Recompute with Otsu's automatic threshold" onClick={() => applyThreshold("otsu")} disabled={applying}>Otsu</Button>
                  <Button variant="ghost" title="Reset slider" onClick={() => setLiveThreshold(detail.result.threshold.value)}><RotateCcw className="h-3.5 w-3.5" /></Button>
                </div>
              </div>
            </>
          )}
        </Panel>
        {detail && (
          <div className="grid gap-4 md:grid-cols-2">
            <HistogramPanel result={detail.result} threshold={liveThreshold} />
            <SpatialGridPanel result={detail.result} />
          </div>
        )}
      </section>

      {/* RIGHT: metrics ------------------------------------------------------------------------- */}
      <aside className="min-w-0 space-y-4 lg:col-start-2 xl:col-start-3 xl:row-start-1">
        {detail ? (
          <>
            <StatsPanel result={detail.result} preview={preview} />
            <WarningsPanel result={detail.result} />
            <InterpretationPanel result={detail.result} onSelectRegion={setSelectedRegion} selectedRegion={selectedRegion} />
            <DownloadsPanel detail={detail} version={version} />
            <EvaluationPanel key={version} detail={detail} sampleGroundTruth={sampleGT} onEvaluated={() => api.get(detail.summary.id).then(setDetail)} />
          </>
        ) : (
          <Panel title="Metrics"><p className="text-sm text-dim">Statistics, warnings and model information appear here after an analysis.</p></Panel>
        )}
        <ModelStatusCard model={deepInfo?.model} />
      </aside>
    </div>
  );
}
