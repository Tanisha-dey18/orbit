import type {
  AnalysisDetail, AnalysisPage, AnalysisParams, EvaluationResult, Health, ModelsResponse, Sample, ThresholdMode,
} from "../types/api";

/** Backend origin. Empty string = same origin (Vite dev proxy). */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  status: number;
  code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network_error",
      "Cannot reach the PROJECT ORBIT backend. Is it running (uvicorn on port 8000) and is VITE_API_BASE_URL correct?");
  }
  if (!res.ok) {
    let code = "http_error";
    let message = `Request failed with HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.error) {
        code = body.error.code ?? code;
        message = body.error.message ?? message;
        if (Array.isArray(body.error.details) && body.error.details.length) {
          message += " " + body.error.details.map((d: { msg: string }) => d.msg).join("; ");
        }
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

export const artifactUrl = (id: string, kind: string, opts?: { download?: boolean; v?: string }) => {
  const q = new URLSearchParams();
  if (opts?.download) q.set("download", "true");
  if (opts?.v) q.set("v", opts.v);
  const qs = q.toString();
  return `${API_BASE}/api/analyses/${id}/artifacts/${kind}${qs ? `?${qs}` : ""}`;
};

export const reportUrl = (id: string) => `${API_BASE}/api/analyses/${id}/report`;
export const sampleUrl = (id: string, kind: "before" | "after" | "ground_truth") => `${API_BASE}/api/samples/${id}/${kind}`;

export const api = {
  health: () => request<Health>("/api/health"),
  models: () => request<ModelsResponse>("/api/models"),
  samples: () => request<{ samples: Sample[] }>("/api/samples").then((r) => r.samples),

  analyze: (before: File, after: File, params: AnalysisParams) => {
    const fd = new FormData();
    fd.append("before", before);
    fd.append("after", after);
    fd.append("params", JSON.stringify(params));
    return request<AnalysisDetail>("/api/analyze", { method: "POST", body: fd });
  },

  list: (page = 1, pageSize = 20, method?: string) => {
    const q = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (method) q.set("method", method);
    return request<AnalysisPage>(`/api/analyses?${q}`);
  },
  get: (id: string) => request<AnalysisDetail>(`/api/analyses/${id}`),
  remove: (id: string) => request<{ id: string; deleted_files: boolean }>(`/api/analyses/${id}`, { method: "DELETE" }),

  rethreshold: (id: string, body: { threshold?: number; threshold_mode: ThresholdMode }) =>
    request<AnalysisDetail>(`/api/analyses/${id}/rethreshold`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  evaluate: (id: string, groundTruth: Blob, filename = "ground_truth.png") => {
    const fd = new FormData();
    fd.append("analysis_id", id);
    fd.append("ground_truth", groundTruth, filename);
    return request<EvaluationResult>("/api/evaluate", { method: "POST", body: fd });
  },

  fetchBlob: async (url: string, name: string): Promise<File> => {
    let res: Response;
    try {
      res = await fetch(url);
    } catch {
      throw new ApiError(0, "network_error", "Cannot reach the backend to download the sample image.");
    }
    if (!res.ok) throw new ApiError(res.status, "http_error", `Could not load ${name} (HTTP ${res.status}).`);
    return new File([await res.blob()], name, { type: "image/png" });
  },
};
