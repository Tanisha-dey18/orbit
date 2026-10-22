import { useRef, useState } from "react";

/** Before/after swipe slider. Renders inside an ImageViewer (absolute, full-size). */
export default function SwipeCompare({ before, after, beforeLabel = "BEFORE", afterLabel = "AFTER" }: {
  before: string; after: string; beforeLabel?: string; afterLabel?: string;
}) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const update = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };

  return (
    <div ref={ref} className="absolute inset-0 select-none">
      <img src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-fill" draggable={false} />
      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-fill" draggable={false} />
      </div>
      <div
        data-no-pan="1"
        role="slider"
        aria-label="Swipe position"
        aria-valuenow={Math.round(pos)}
        aria-valuemin={0}
        aria-valuemax={100}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 2));
          if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 2));
        }}
        onPointerDown={(e) => { e.stopPropagation(); dragging.current = true; (e.target as HTMLElement).setPointerCapture(e.pointerId); }}
        onPointerMove={(e) => dragging.current && update(e.clientX)}
        onPointerUp={() => { dragging.current = false; }}
        className="absolute inset-y-0 z-10 -ml-3 w-6 cursor-ew-resize"
        style={{ left: `${pos}%` }}
      >
        <div data-no-pan="1" className="mx-auto h-full w-px bg-cyan shadow-[0_0_8px_rgb(34_211_238)]" />
        <div data-no-pan="1" className="absolute left-1/2 top-1/2 h-7 w-7 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan bg-space-950/90 text-center font-mono text-[10px] leading-7 text-cyan">⇔</div>
      </div>
      <span className="absolute left-2 top-2 rounded bg-space-950/80 px-1.5 py-0.5 font-mono text-[10px] text-ink">{beforeLabel}</span>
      <span className="absolute bottom-2 right-2 rounded bg-space-950/80 px-1.5 py-0.5 font-mono text-[10px] text-ink">{afterLabel}</span>
    </div>
  );
}
