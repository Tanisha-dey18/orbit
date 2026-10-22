export const fmtInt = (n: number | null | undefined) => (n == null ? "–" : n.toLocaleString("en-US"));

export const fmtPct = (n: number | null | undefined, digits = 2) => (n == null ? "–" : `${n.toFixed(digits)}%`);

export const fmtMs = (ms: number | null | undefined) => {
  if (ms == null) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
};

export const fmtBytes = (b: number) => {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(2)} MB`;
};

export const fmtArea = (m2: number | null | undefined) => {
  if (m2 == null) return null;
  return m2 >= 1e6 ? `${(m2 / 1e6).toFixed(3)} km²` : `${Math.round(m2).toLocaleString("en-US")} m²`;
};

export const fmtDate = (iso: string) => {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};

export const fmtMetric = (v: number | null | undefined) => (v == null ? "undefined" : v.toFixed(4));

export const shortId = (id: string) => id.slice(0, 8);
