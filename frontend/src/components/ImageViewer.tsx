import { Maximize2, Minus, Plus } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

const clampZoom = (z: number) => Math.min(8, Math.max(1, z));

/**
 * Zoom/pan container. Children are rendered in image-pixel space (width x height) and
 * scaled together, so overlays (canvas, region boxes) stay registered with the image.
 */
export default function ImageViewer({ width, height, children, caption }: {
  width: number; height: number; children: ReactNode; caption?: ReactNode;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);

  const reset = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  // Native listener: React's onWheel is passive, so preventDefault() would be ignored.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      setZoom((z) => {
        const nz = clampZoom(z * (e.deltaY < 0 ? 1.2 : 1 / 1.2));
        const k = nz / z;
        setPan((p) => (nz === 1 ? { x: 0, y: 0 } : { x: cx - (cx - p.x) * k, y: cy - (cy - p.y) * k }));
        return nz;
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div className="relative">
      <div
        ref={box}
        className="checker relative mx-auto overflow-hidden rounded-lg border border-line"
        style={{ aspectRatio: `${width} / ${height}`, width: `min(100%, calc(70vh * ${width / height}))`, cursor: zoom > 1 ? (drag.current ? "grabbing" : "grab") : "default" }}
        onPointerDown={(e) => {
          if (zoom === 1 || (e.target as HTMLElement).dataset.noPan) return;
          drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          setPan({ x: drag.current.px + e.clientX - drag.current.x, y: drag.current.py + e.clientY - drag.current.y });
        }}
        onPointerUp={() => { drag.current = null; }}
        onDoubleClick={reset}
      >
        <div
          className="absolute inset-0 origin-center"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transition: drag.current ? "none" : "transform 120ms ease-out" }}
        >
          {children}
        </div>
      </div>
      <div className="absolute right-2 top-2 z-10 flex items-center gap-1 rounded-md border border-line bg-space-950/80 p-0.5 backdrop-blur">
        <button className="rounded p-1 text-muted hover:text-cyan" onClick={() => setZoom((z) => clampZoom(z / 1.4))} aria-label="Zoom out"><Minus className="h-3.5 w-3.5" /></button>
        <span className="w-10 text-center font-mono text-[10px] text-muted">{Math.round(zoom * 100)}%</span>
        <button className="rounded p-1 text-muted hover:text-cyan" onClick={() => setZoom((z) => clampZoom(z * 1.4))} aria-label="Zoom in"><Plus className="h-3.5 w-3.5" /></button>
        <button className="rounded p-1 text-muted hover:text-cyan" onClick={reset} aria-label="Reset view"><Maximize2 className="h-3.5 w-3.5" /></button>
      </div>
      {caption && <div className="mt-2 text-xs text-muted">{caption}</div>}
    </div>
  );
}
