import { ChevronLeft, ChevronRight, ExternalLink, History, RefreshCw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Badge, Button, EmptyState, Notice, Panel, Spinner } from "../components/ui";
import { api } from "../services/api";
import type { AnalysisPage } from "../types/api";
import { fmtDate, fmtMs, fmtPct, shortId } from "../utils/format";

const PAGE_SIZE = 15;

export default function HistoryPage() {
  const [page, setPage] = useState(1);
  const [method, setMethod] = useState<string>("");
  const [data, setData] = useState<AnalysisPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.list(page, PAGE_SIZE, method || undefined));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [page, method]);

  useEffect(() => { load(); }, [load]);

  const remove = async (id: string) => {
    setDeleting(true);
    try {
      await api.remove(id);
      setConfirmId(null);
      if (data && data.items.length === 1 && page > 1) setPage(page - 1);
      else load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  };

  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="label">Local analysis records (SQLite)</p>
          <h1 className="font-display text-2xl font-semibold">Analysis history</h1>
        </div>
        <div className="flex items-center gap-2">
          <select value={method} onChange={(e) => { setMethod(e.target.value); setPage(1); }}
            className="rounded-md border border-line bg-space-900 px-2 py-1.5 text-sm">
            <option value="">All methods</option>
            <option value="classical">Classical baseline</option>
            <option value="deep">Deep learning</option>
          </select>
          <Button onClick={load} disabled={loading}><RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh</Button>
        </div>
      </div>
      {error && <div className="mb-4"><Notice tone="error">{error}</Notice></div>}
      <Panel className="overflow-hidden !p-0">
        {loading && !data ? (
          <div className="flex justify-center py-16"><Spinner className="h-6 w-6 text-cyan" /></div>
        ) : !data || data.items.length === 0 ? (
          <EmptyState icon={<History className="h-6 w-6" />} title="No analyses recorded">
            Run an analysis from the <Link to="/analyze" className="text-cyan hover:underline">Analyze</Link> page and it will appear here.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] text-sm">
              <thead className="border-b border-line bg-space-900/70">
                <tr className="label text-left">
                  {["ID", "Created", "Inputs", "Method", "Changed", "Regions", "Status", "Time", ""].map((h) => (
                    <th key={h} className="px-3 py-2.5 font-normal">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.items.map((a) => (
                  <tr key={a.id} className="border-b border-line/60 hover:bg-space-800/50">
                    <td className="px-3 py-2.5 font-mono text-xs text-cyan"><Link to={`/analysis/${a.id}`}>{shortId(a.id)}</Link></td>
                    <td className="px-3 py-2.5 text-xs text-muted">{fmtDate(a.created_at)}</td>
                    <td className="max-w-[220px] px-3 py-2.5 text-xs"><span className="block truncate" title={`${a.before_filename} → ${a.after_filename}`}>{a.before_filename} → {a.after_filename}</span></td>
                    <td className="px-3 py-2.5">
                      {a.method === "deep" ? <Badge tone="violet">deep · {a.model_version ?? "?"}</Badge> : <Badge tone="cyan">classical · {a.method_variant}</Badge>}
                    </td>
                    <td className="px-3 py-2.5 font-mono text-xs">{fmtPct(a.changed_pixel_percentage, 3)}</td>
                    <td className="px-3 py-2.5 font-mono text-xs">{a.region_count}</td>
                    <td className="px-3 py-2.5">
                      <Badge tone={a.status === "unreliable" ? "change" : a.status === "completed" ? "ok" : "warn"}>{a.status.replace(/_/g, " ")}</Badge>
                    </td>
                    <td className="px-3 py-2.5 font-mono text-xs text-muted">{fmtMs(a.total_time_ms)}</td>
                    <td className="px-3 py-2.5">
                      {confirmId === a.id ? (
                        <span className="flex items-center gap-1.5">
                          <Button variant="danger" className="!px-2 !py-1 text-xs" disabled={deleting} onClick={() => remove(a.id)}>{deleting ? <Spinner /> : "Delete"}</Button>
                          <Button variant="ghost" className="!px-2 !py-1 text-xs" onClick={() => setConfirmId(null)}>Cancel</Button>
                        </span>
                      ) : (
                        <span className="flex items-center gap-1">
                          <Link to={`/analysis/${a.id}`} className="rounded p-1.5 text-muted hover:text-cyan" aria-label="Open"><ExternalLink className="h-3.5 w-3.5" /></Link>
                          <button className="rounded p-1.5 text-muted hover:text-change" aria-label="Delete" onClick={() => setConfirmId(a.id)}><Trash2 className="h-3.5 w-3.5" /></button>
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      {data && data.total > 0 && (
        <div className="mt-3 flex items-center justify-between text-xs text-muted">
          <span>{data.total} record(s) · deleting removes the database record and all generated files for that analysis.</span>
          <span className="flex items-center gap-2">
            <Button variant="ghost" disabled={page <= 1} onClick={() => setPage(page - 1)}><ChevronLeft className="h-4 w-4" /></Button>
            page {page} / {pages}
            <Button variant="ghost" disabled={page >= pages} onClick={() => setPage(page + 1)}><ChevronRight className="h-4 w-4" /></Button>
          </span>
        </div>
      )}
    </div>
  );
}
