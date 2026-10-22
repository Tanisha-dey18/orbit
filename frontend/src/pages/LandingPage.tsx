import {
  ArrowRight, Brain, Crosshair, Database, FileText, GitCompareArrows, Layers, Radar, ScanSearch, ShieldCheck, Waves,
} from "lucide-react";
import { Link } from "react-router-dom";
import ImageViewer from "../components/ImageViewer";
import SwipeCompare from "../components/SwipeCompare";
import { Badge } from "../components/ui";

function HeroVisual() {
  // Illustrative graphic only - not imagery.
  return (
    <svg viewBox="0 0 520 420" className="h-full w-full" role="img" aria-label="Illustration of a satellite orbiting Earth and scanning a region">
      <defs>
        <radialGradient id="earth" cx="40%" cy="35%" r="70%">
          <stop offset="0%" stopColor="#12355e" />
          <stop offset="60%" stopColor="#0a1f3b" />
          <stop offset="100%" stopColor="#050d1c" />
        </radialGradient>
        <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </linearGradient>
        <clipPath id="globe"><circle cx="260" cy="250" r="150" /></clipPath>
      </defs>
      <circle cx="260" cy="250" r="168" fill="none" stroke="#22d3ee" strokeOpacity="0.08" strokeWidth="18" />
      <circle cx="260" cy="250" r="150" fill="url(#earth)" stroke="#1c2c45" />
      <g clipPath="url(#globe)" stroke="#22d3ee" strokeOpacity="0.14" fill="none">
        {[-100, -50, 0, 50, 100].map((d) => <ellipse key={d} cx="260" cy="250" rx={Math.abs(150 - Math.abs(d) * 0.9) || 4} ry="150" />)}
        {[-110, -60, 0, 60, 110].map((d) => <line key={d} x1="100" x2="420" y1={250 + d} y2={250 + d} />)}
        <path d="M150 210 q40 -30 80 -10 t70 20 q30 10 60 -15" stroke="#34d399" strokeOpacity="0.35" strokeWidth="10" />
        <path d="M170 300 q50 20 100 0 t90 15" stroke="#3b82f6" strokeOpacity="0.4" strokeWidth="8" />
      </g>
      <g className="orbit-spin">
        <ellipse cx="260" cy="250" rx="235" ry="78" fill="none" stroke="#a78bfa" strokeOpacity="0.35" transform="rotate(-18 260 250)" strokeDasharray="2 6" />
      </g>
      <ellipse cx="260" cy="250" rx="235" ry="78" fill="none" stroke="#22d3ee" strokeOpacity="0.25" transform="rotate(-18 260 250)" />
      <polygon points="372,96 318,246 386,246" fill="url(#beam)" />
      <rect x="328" y="222" width="52" height="36" fill="none" stroke="#ff4060" strokeWidth="1.5" strokeDasharray="4 3" />
      <g transform="translate(372 92) rotate(-18)">
        <rect x="-26" y="-6" width="18" height="12" fill="#3b82f6" stroke="#22d3ee" strokeWidth="0.8" />
        <rect x="8" y="-6" width="18" height="12" fill="#3b82f6" stroke="#22d3ee" strokeWidth="0.8" />
        <rect x="-8" y="-9" width="16" height="18" rx="2" fill="#e6eef8" />
        <circle cx="0" cy="9" r="3" fill="#22d3ee" />
      </g>
      <text x="390" y="275" fill="#ff4060" fontFamily="JetBrains Mono" fontSize="10">Δ change region</text>
    </svg>
  );
}

const CAPABILITIES = [
  { icon: <Crosshair className="h-5 w-5" />, title: "Registration & normalisation", text: "ORB + RANSAC and ECC alignment, geospatial reprojection for GeoTIFFs, radiometric matching, and explicit failure detection." },
  { icon: <GitCompareArrows className="h-5 w-5" />, title: "Classical baseline", text: "Change Vector Analysis and grayscale differencing with Otsu or manual thresholds, morphology and connected components." },
  { icon: <Brain className="h-5 w-5" />, title: "Siamese U-Net inference", text: "FC-Siam-diff architecture with tiled CPU/GPU inference, verified checkpoint loading and a full LEVIR-CD training pipeline." },
  { icon: <Waves className="h-5 w-5" />, title: "Disaster intelligence", text: "Conservative, rule-based interpretation of infrastructure, building, vegetation and potential flood-related change - with evidence." },
  { icon: <ScanSearch className="h-5 w-5" />, title: "Interactive inspection", text: "Swipe comparison, heatmaps, masks, adjustable overlays, live threshold preview, zoom/pan and region highlighting." },
  { icon: <FileText className="h-5 w-5" />, title: "Reports & evaluation", text: "PDF reports, PNG/JSON exports, and precision / recall / F1 / IoU against ground-truth masks." },
];

const STACK = [
  ["Computer vision", "OpenCV · NumPy · Rasterio (optional)"],
  ["Deep learning", "PyTorch · FC-Siam-diff (Siamese U-Net)"],
  ["Backend", "FastAPI · Pydantic · SQLAlchemy · SQLite · ReportLab"],
  ["Frontend", "React · TypeScript · Vite · Tailwind · Recharts"],
];

export default function LandingPage() {
  return (
    <div>
      <section className="mx-auto grid max-w-7xl items-center gap-10 px-4 pb-10 pt-12 lg:grid-cols-[1.05fr_1fr] lg:pt-20">
        <div>
          <p className="label !text-cyan">Satellite image change detection · disaster intelligence</p>
          <h1 className="mt-3 font-display text-5xl font-bold leading-[1.05] tracking-tight md:text-6xl">
            See what changed<br /> <span className="bg-gradient-to-r from-cyan to-electric bg-clip-text text-transparent">between two passes.</span>
          </h1>
          <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-muted">
            PROJECT ORBIT compares satellite images of the same region captured at different times. It aligns them,
            detects pixel-level change with a classical baseline or a Siamese U-Net, quantifies it, and helps you
            investigate urban growth, infrastructure, vegetation loss and potential flood-related change - with
            honest reporting of uncertainty.
          </p>
          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Link to="/analyze" className="inline-flex items-center gap-2 rounded-lg bg-cyan px-5 py-3 font-semibold text-space-950 shadow-[0_0_32px_-8px_rgb(34_211_238/0.8)] transition-colors hover:bg-cyan/90">
              Launch Analysis <ArrowRight className="h-4 w-4" />
            </Link>
            <Link to="/methodology" className="rounded-lg border border-line px-5 py-3 text-sm text-ink hover:border-cyan/50 hover:text-cyan">How it works</Link>
          </div>
          <p className="mt-6 flex items-center gap-2 text-xs text-dim">
            <ShieldCheck className="h-4 w-4 text-ok" /> A visual change is evidence to investigate, not proof of a disaster.
          </p>
        </div>
        <div className="relative mx-auto aspect-[52/42] w-full max-w-xl"><HeroVisual /></div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-10">
        <div className="grid gap-8 lg:grid-cols-[1fr_1.1fr] lg:items-center">
          <div>
            <p className="label">Sample comparison</p>
            <h2 className="mt-2 font-display text-3xl font-semibold">Drag to compare</h2>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              This is a <span className="text-violet">synthetic, procedurally generated</span> scene bundled for demonstration - not real
              satellite imagery. Its AFTER image contains new buildings and a road, plus a deliberate 1.2° rotation, a few pixels
              of shift, an illumination change and sensor noise, so the alignment step has real work to do.
            </p>
            <p className="mt-3 text-sm text-muted">Load it (and two other scenarios) from <span className="text-cyan">Analyze → Sample pair</span>, with exact ground truth for evaluation.</p>
          </div>
          <div className="panel p-3">
            <ImageViewer width={512} height={512} caption={<span className="flex items-center gap-2"><Badge tone="violet">synthetic</Badge> Illustrative scene generated by app/utils/synthetic.py</span>}>
              <SwipeCompare before="/samples/urban_expansion_before.png" after="/samples/urban_expansion_after.png" />
            </ImageViewer>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-10">
        <p className="label">Capabilities</p>
        <h2 className="mt-2 font-display text-3xl font-semibold">A real pipeline, end to end</h2>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {CAPABILITIES.map((c) => (
            <div key={c.title} className="panel p-5 transition-colors hover:border-cyan/40">
              <div className="mb-3 inline-flex rounded-lg border border-line bg-space-900 p-2 text-cyan">{c.icon}</div>
              <h3 className="font-display text-lg font-semibold">{c.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{c.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pb-16 pt-6">
        <div className="panel grid gap-6 p-6 md:grid-cols-[1fr_1.4fr]">
          <div>
            <p className="label">Technology</p>
            <h2 className="mt-2 font-display text-2xl font-semibold">Built to run on a laptop</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Everything runs locally, CPU-only if needed, with no paid APIs. The classical baseline always works; deep-learning
              mode activates only when a verified checkpoint is present.
            </p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs text-muted">
              <span className="flex items-center gap-1"><Database className="h-3.5 w-3.5 text-cyan" /> SQLite history</span>
              <span className="flex items-center gap-1"><Layers className="h-3.5 w-3.5 text-cyan" /> local artifacts</span>
              <span className="flex items-center gap-1"><Radar className="h-3.5 w-3.5 text-cyan" /> REST API</span>
            </div>
          </div>
          <dl className="grid gap-2 sm:grid-cols-2">
            {STACK.map(([k, v]) => (
              <div key={k} className="rounded-lg border border-line bg-space-900/60 p-3">
                <dt className="label">{k}</dt>
                <dd className="mt-1 text-sm text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>
    </div>
  );
}
