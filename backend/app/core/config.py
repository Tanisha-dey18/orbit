"""Application configuration.

All settings are read from environment variables prefixed with ``ORBIT_`` (or a
``.env`` file). Defaults are chosen so the app runs locally with zero setup.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repo root is three parents above backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ORBIT_",
        env_file=(str(REPO_ROOT / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PROJECT ORBIT API"
    environment: str = "development"

    database_url: str = Field(default=f"sqlite:///{(REPO_ROOT / 'artifacts' / 'orbit.db').as_posix()}")
    artifact_dir: Path = Field(default=REPO_ROOT / "artifacts")
    model_path: Path = Field(default=REPO_ROOT / "data" / "checkpoints" / "orbit_fc_siam_diff.pt")
    samples_dir: Path = Field(default=REPO_ROOT / "data" / "samples")

    max_upload_mb: float = 25.0
    # Images larger than this (longest side, pixels) are downscaled for processing.
    max_processing_side: int = 2048
    # Hard ceiling on decoded pixels per image to protect memory (decompression bombs).
    max_image_pixels: int = 100_000_000
    min_image_side: int = 32
    max_batch_pairs: int = 5
    max_concurrent_analyses: int = 2

    # Comma-separated list. Use explicit origins in production, never "*".
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    device: str = "auto"  # auto | cpu | cuda
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("artifact_dir", "model_path", "samples_dir", mode="before")
    @classmethod
    def _resolve_relative(cls, v):
        p = Path(v)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @field_validator("database_url")
    @classmethod
    def _resolve_sqlite_path(cls, v: str) -> str:
        # Make relative SQLite paths independent of the directory uvicorn is started from.
        prefix = "sqlite:///"
        if v.startswith(prefix):
            path = v[len(prefix):]
            if path and path != ":memory:" and not Path(path).is_absolute():
                return prefix + (REPO_ROOT / path).resolve().as_posix()
        return v

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
