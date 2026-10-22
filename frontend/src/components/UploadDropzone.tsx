import { ImageUp, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { fmtBytes } from "../utils/format";

const ACCEPT = ".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff";

export interface LocalImage {
  file: File;
  previewUrl: string | null; // null for TIFF (browsers can't render it)
  width?: number;
  height?: number;
  synthetic?: boolean;
}

export function useLocalImage(file: File | null, synthetic = false): LocalImage | null {
  const [info, setInfo] = useState<LocalImage | null>(null);
  useEffect(() => {
    if (!file) {
      setInfo(null);
      return;
    }
    const isTiff = /\.tiff?$/i.test(file.name) || file.type === "image/tiff";
    if (isTiff) {
      setInfo({ file, previewUrl: null, synthetic });
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => setInfo({ file, previewUrl: url, width: img.naturalWidth, height: img.naturalHeight, synthetic });
    img.onerror = () => setInfo({ file, previewUrl: null, synthetic });
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file, synthetic]);
  return info;
}

export default function UploadDropzone({ label, image, onFile, maxMb }: {
  label: string; image: LocalImage | null; onFile: (f: File | null) => void; maxMb: number;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accept = (f: File | undefined) => {
    if (!f) return;
    if (!/\.(png|jpe?g|tiff?)$/i.test(f.name)) {
      setError("Unsupported file type. Use PNG, JPEG or TIFF.");
      return;
    }
    if (f.size > maxMb * 1024 * 1024) {
      setError(`File is ${fmtBytes(f.size)}; the limit is ${maxMb} MB.`);
      return;
    }
    setError(null);
    onFile(f);
  };

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="label">{label}</span>
        {image && (
          <button className="text-dim hover:text-change" onClick={() => onFile(null)} aria-label={`Clear ${label}`}>
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); accept(e.dataTransfer.files[0]); }}
        className={`group relative flex h-28 w-full items-center justify-center overflow-hidden rounded-lg border border-dashed transition-colors ${
          drag ? "border-cyan bg-cyan/10" : "border-space-600 bg-space-900/60 hover:border-cyan/60"
        }`}
      >
        {image?.previewUrl ? (
          <img src={image.previewUrl} alt={`${label} preview`} className="h-full w-full object-cover opacity-90" />
        ) : image ? (
          <span className="px-2 text-xs text-muted">TIFF selected (no browser preview)</span>
        ) : (
          <span className="flex flex-col items-center gap-1 text-xs text-muted">
            <ImageUp className="h-5 w-5 text-cyan/80" />
            Drop or click · PNG / JPEG / TIFF
          </span>
        )}
        {image?.synthetic && (
          <span className="absolute left-1.5 top-1.5 rounded bg-violet/90 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-space-950">
            SYNTHETIC
          </span>
        )}
      </button>
      <input ref={input} type="file" accept={ACCEPT} className="hidden" onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }} />
      {image && (
        <p className="mt-1 truncate font-mono text-[10px] text-dim" title={image.file.name}>
          {image.file.name} · {fmtBytes(image.file.size)}
          {image.width ? ` · ${image.width}×${image.height}` : ""}
        </p>
      )}
      {error && <p className="mt-1 text-[11px] text-change">{error}</p>}
    </div>
  );
}
