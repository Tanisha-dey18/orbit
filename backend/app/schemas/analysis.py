"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Method = Literal["classical", "deep"]
ClassicalMethod = Literal["cva", "grayscale_diff"]
ThresholdMode = Literal["manual", "otsu"]
AlignmentMethod = Literal["auto", "feature", "ecc", "none"]
Normalization = Literal["mean_std", "histogram", "none"]
Context = Literal["general", "urban", "flood", "vegetation", "post_disaster"]

DEFAULT_THRESHOLDS = {"classical": 0.12, "deep": 0.5}


class AnalysisParams(BaseModel):
    """Parameters for one analysis. Sent as a JSON string in the multipart `params` field."""

    model_config = ConfigDict(extra="forbid")

    method: Method = "classical"
    classical_method: ClassicalMethod = "cva"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0,
                                    description="Change-intensity/probability threshold in [0,1]. "
                                                "Defaults: classical 0.12, deep 0.5.")
    threshold_mode: ThresholdMode = "manual"
    alignment: AlignmentMethod = "auto"
    normalization: Normalization = "mean_std"
    blur_sigma: float = Field(default=1.0, ge=0.0, le=5.0)
    morph_kernel: int = Field(default=3, ge=0, le=15)
    min_region_px: int = Field(default=16, ge=0, le=100_000)
    context: Context = "general"

    @field_validator("morph_kernel")
    @classmethod
    def _odd_kernel(cls, v: int) -> int:
        return v if v <= 1 or v % 2 == 1 else v + 1

    def effective_threshold(self) -> float:
        return self.threshold if self.threshold is not None else DEFAULT_THRESHOLDS[self.method]


class RethresholdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_mode: ThresholdMode = "manual"
    morph_kernel: int | None = Field(default=None, ge=0, le=15)
    min_region_px: int | None = Field(default=None, ge=0, le=100_000)


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    before_filename: str
    after_filename: str
    method: str
    method_variant: str | None
    model_name: str | None
    model_version: str | None
    status: str
    context: str | None
    threshold: float
    changed_pixel_count: int
    valid_pixel_count: int
    changed_pixel_percentage: float
    region_count: int
    changed_area_m2: float | None
    inference_time_ms: float
    total_time_ms: float
    warning_summary: str | None

    @field_validator("created_at")
    @classmethod
    def _as_utc(cls, v: datetime) -> datetime:
        # SQLite drops tzinfo; values are always stored in UTC.
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v


class AnalysisPage(BaseModel):
    items: list[AnalysisSummary]
    total: int
    page: int
    page_size: int


class ArtifactInfo(BaseModel):
    artifact_type: str
    media_type: str
    size_bytes: int
    created_at: datetime
    url: str

    @field_validator("created_at")
    @classmethod
    def _as_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v


class AnalysisDetail(BaseModel):
    summary: AnalysisSummary
    result: dict[str, Any]
    artifacts: list[ArtifactInfo]


class BatchItemResult(BaseModel):
    index: int
    ok: bool
    analysis_id: str | None = None
    changed_pixel_percentage: float | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    results: list[BatchItemResult]


class EvaluationResponse(BaseModel):
    analysis_id: str
    threshold: float
    method: str
    evaluated_pixels: int
    excluded_pixels: int
    metrics: dict[str, Any]
    warnings: list[str]


class DeleteResponse(BaseModel):
    id: str
    deleted_records: int
    deleted_files: bool
