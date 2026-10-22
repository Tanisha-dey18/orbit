import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import MethodBadge from "../components/MethodBadge";
import { ModelStatusCard } from "../components/ResultPanels";
import AnalyzePage from "../pages/AnalyzePage";
import HistoryPage from "../pages/HistoryPage";
import { api, ApiError } from "../services/api";
import type { AnalysisResult, ModelInfo } from "../types/api";
import { fmtArea, fmtMetric, fmtPct } from "../utils/format";
import { thresholdToOverlay } from "../utils/threshold";

afterEach(() => vi.restoreAllMocks());

describe("utils", () => {
  it("formats values without inventing numbers", () => {
    expect(fmtPct(1.23456, 2)).toBe("1.23%");
    expect(fmtArea(null)).toBeNull();
    expect(fmtArea(2_500_000)).toBe("2.500 km²");
    expect(fmtMetric(null)).toBe("undefined");
  });

  it("thresholds an intensity image client-side", () => {
    const img = new ImageData(2, 2);
    img.data.set([10, 10, 10, 255, 200, 200, 200, 255, 0, 0, 0, 255, 128, 128, 128, 255]);
    expect(thresholdToOverlay(img, 0.5).changed).toBe(2);
    expect(thresholdToOverlay(img, 0.9).changed).toBe(0);
  });
});

describe("api client", () => {
  it("maps network failures to a helpful ApiError", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.health()).rejects.toMatchObject({ code: "network_error" });
  });

  it("surfaces backend error messages", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "model_unavailable", message: "No checkpoint" } }), { status: 409 }),
    );
    const err = await api.models().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.message).toBe("No checkpoint");
  });
});

const baseResult = (trained: boolean, status: AnalysisResult["status"] = "completed") =>
  ({ method: { is_trained_model: trained, model: trained ? { dataset: "LEVIR-CD" } : null }, status }) as unknown as AnalysisResult;

describe("labelling", () => {
  it("distinguishes classical baseline from model output", () => {
    const { rerender } = render(<MethodBadge result={baseResult(false)} />);
    expect(screen.getByText(/Classical baseline/i)).toBeInTheDocument();
    rerender(<MethodBadge result={baseResult(true, "unreliable")} />);
    expect(screen.getByText(/Deep model output/i)).toBeInTheDocument();
    expect(screen.getByText(/unreliable comparison/i)).toBeInTheDocument();
  });

  it("shows missing model state", () => {
    const model: ModelInfo = { status: "missing", ready: false, message: "No trained checkpoint found", device: "cpu", metadata: null };
    render(<ModelStatusCard model={model} />);
    expect(screen.getByText(/status: missing/)).toBeInTheDocument();
    expect(screen.getByText(/No trained checkpoint found/)).toBeInTheDocument();
  });
});

describe("pages", () => {
  it("analyze page: offline backend disables running and explains why", () => {
    render(
      <MemoryRouter initialEntries={["/analyze"]}>
        <Routes><Route path="/analyze" element={<AnalyzePage backend={{ kind: "offline", error: "down" }} />} /></Routes>
      </MemoryRouter>,
    );
    expect(screen.getByRole("button", { name: /Run analysis/i })).toBeDisabled();
    expect(screen.getByText(/Backend offline/i)).toBeInTheDocument();
    expect(screen.getByText(/No analysis yet/i)).toBeInTheDocument();
  });

  it("history page: empty state", async () => {
    vi.spyOn(api, "list").mockResolvedValue({ items: [], total: 0, page: 1, page_size: 15 });
    render(<MemoryRouter><HistoryPage /></MemoryRouter>);
    expect(await screen.findByText(/No analyses recorded/i)).toBeInTheDocument();
  });

  it("history page: API failure is shown", async () => {
    vi.spyOn(api, "list").mockRejectedValue(new ApiError(0, "network_error", "Cannot reach the PROJECT ORBIT backend"));
    render(<MemoryRouter><HistoryPage /></MemoryRouter>);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/Cannot reach/));
  });
});
