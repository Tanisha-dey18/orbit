"""ORM models. Image binaries are never stored in the database - only file names."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    before_filename: Mapped[str] = mapped_column(String(255))
    after_filename: Mapped[str] = mapped_column(String(255))
    method: Mapped[str] = mapped_column(String(32))  # "classical" | "deep"
    method_variant: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32))  # completed | completed_with_warnings | unreliable
    context: Mapped[str | None] = mapped_column(String(32), nullable=True)
    threshold: Mapped[float] = mapped_column(Float)
    changed_pixel_count: Mapped[int] = mapped_column(Integer)
    valid_pixel_count: Mapped[int] = mapped_column(Integer)
    changed_pixel_percentage: Mapped[float] = mapped_column(Float)
    region_count: Mapped[int] = mapped_column(Integer, default=0)
    changed_area_m2: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float] = mapped_column(Float)
    total_time_ms: Mapped[float] = mapped_column(Float)
    warning_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)

    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", passive_deletes=True
    )


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(32))
    stored_filename: Mapped[str] = mapped_column(String(128))
    media_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    analysis: Mapped[Analysis] = relationship(back_populates="artifacts")
