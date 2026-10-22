// Types mirroring the FastAPI schemas (backend/app/schemas/analysis.py).

export type Method = "classical" | "deep";
export type ClassicalMethod = "cva" | "grayscale_diff";
export type ThresholdMode = "manual" | "otsu";
export type AlignmentMethod = "auto" | "feature" | "ecc" | "none";
export type Normalization = "mean_std" | "histogram" | "none";
export type Context = "general" | "urban" | "flood" | "vegetation" | "post_disaster";

export interface AnalysisParams {
  method: Method;
  classical_method: ClassicalMethod;
  threshold: number | null;
  threshold_mode: ThresholdMode;
  alignment: AlignmentMethod;
  normalization: Normalization;
  blur_sigma: number;
  morph_kernel: number;
  min_region_px: number;
  context: Context;
}

export interface ModelInfo {
  status: "loaded" | "missing" | "invalid" | "not_loaded";
  ready: boolean;
  message: string;
  device: string;
  metadata: Record<string, unknown> | null;
}

export interface Health {
  status: string;
  version: string;
  backend: string;
  model: { status: string; ready: boolean };
  capabilities: { classical: boolean; deep_learning: boolean; geotiff_georeferencing: boolean };
}

export interface MethodInfo {
  id: Method;
  label: string;
  available: boolean;
  is_trained_model: boolean;
  default_threshold: number;
  description: string;
  variants?: { id: ClassicalMethod; label: string }[];
  model?: ModelInfo;
}

export interface ModelsResponse {
  methods: MethodInfo[];
  contexts: Context[];
  limits: { max_upload_mb: number; max_processing_side: number; max_batch_pairs: number };
}

export interface Sample {
  id: string;
  title: string;
  description: string;
  synthetic: boolean;
  source: string;
  width: number;
  height: number;
  has_ground_truth: boolean;
}

export interface Region {
  id: number;
  bbox: [number, number, number, number];
  area_px: number;
  share_of_change: number;
  centroid: [number, number];
  mean_intensity?: number;
}

export interface Statistics {
  valid_pixel_count: number;
  changed_pixel_count: number;
  changed_pixel_percentage: number;
  total_pixel_count: number;
  region_count: number;
  largest_region_px: number;
  regions: Region[];
  mean_intensity_changed: number;
  mean_intensity_all: number;
  intensity_p95: number;
  spatial_grid_pct: number[][];
}

export interface CategorySummary {
  label: string;
  regions: number;
  pixels: number;
  share_of_change: number;
}

export interface InterpretedRegion {
  id: number;
  bbox: [number, number, number, number];
  area_px: number;
  category: string;
  confidence: string;
  evidence: string[];
}

export interface Interpretation {
  context: Context;
  categories: Record<string, CategorySummary>;
  regions: InterpretedRegion[];
  intensity_level: string;
  spatial_distribution: string;
  inspection_targets: { region_id: number; bbox: number[]; area_px: number; reason: string }[];
  summary: string[];
  notice: string;
}

export interface ImageMeta {
  filename: string;
  format: string;
  width: number;
  height: number;
  bands: number;
  dtype: string;
  georeferenced: boolean;
  crs: string | null;
}

export interface EvaluationResult {
  analysis_id: string;
  threshold: number;
  method: string;
  evaluated_pixels: number;
  excluded_pixels: number;
  metrics: {
    tp: number; fp: number; fn: number; tn: number;
    precision: number | null; recall: number | null; f1: number | null; iou: number | null; accuracy: number | null;
  };
  warnings: string[];
}

export interface AnalysisResult {
  id: string;
  created_at: string;
  status: "completed" | "completed_with_warnings" | "unreliable";
  reliable: boolean;
  inputs: { before: ImageMeta; after: ImageMeta };
  processed_size: [number, number];
  method: {
    type: Method;
    variant: string;
    label: string;
    is_trained_model: boolean;
    description: string;
    model: Record<string, unknown> | null;
  };
  parameters: AnalysisParams;
  threshold: { value: number; mode: ThresholdMode };
  alignment: Record<string, unknown> & { status: string; method_used: string; valid_fraction: number };
  statistics: Statistics;
  area: { changed_area_m2: number | null; changed_area_km2: number | null; explanation: string; crs: string | null };
  interpretation: Interpretation;
  histogram: { bin_start: number; count: number }[];
  timing: { inference_ms: number; total_ms?: number };
  warnings: string[];
  limitations: string[];
  evaluation: EvaluationResult | null;
  rethresholded_at?: string;
}

export interface AnalysisSummary {
  id: string;
  created_at: string;
  before_filename: string;
  after_filename: string;
  method: Method;
  method_variant: string | null;
  model_name: string | null;
  model_version: string | null;
  status: string;
  context: string | null;
  threshold: number;
  changed_pixel_count: number;
  valid_pixel_count: number;
  changed_pixel_percentage: number;
  region_count: number;
  changed_area_m2: number | null;
  inference_time_ms: number;
  total_time_ms: number;
  warning_summary: string | null;
}

export interface ArtifactInfo {
  artifact_type: string;
  media_type: string;
  size_bytes: number;
  created_at: string;
  url: string;
}

export interface AnalysisDetail {
  summary: AnalysisSummary;
  result: AnalysisResult;
  artifacts: ArtifactInfo[];
}

export interface AnalysisPage {
  items: AnalysisSummary[];
  total: number;
  page: number;
  page_size: number;
}
