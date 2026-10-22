"""PROJECT ORBIT - FastAPI application entry point.

Run locally:  uvicorn app.main:app --reload --port 8000   (from the backend/ folder)
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyses, system
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.database.session import init_db
from app.services.deep import model_service

logger = logging.getLogger("orbit")


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    s.artifact_dir.mkdir(parents=True, exist_ok=True)
    init_db(s.database_url)
    model_service.load(s.model_path, s.device)
    logger.info("PROJECT ORBIT API ready (model status: %s)", model_service.status)
    yield


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    app = FastAPI(
        title=s.app_name,
        version=system.VERSION,
        description="Satellite image change detection: classical baseline + Siamese U-Net inference.",
        lifespan=lifespan,
    )
    if "*" in s.origins:
        logger.warning("ORBIT_ALLOWED_ORIGINS contains '*'; restrict it in production.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = rid
        logging.getLogger("orbit.access").info(
            "%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, ms,
            extra={"request_id": rid, "method": request.method, "path": request.url.path,
                   "status": response.status_code, "duration_ms": round(ms, 1)})
        return response

    register_exception_handlers(app)
    app.include_router(system.router)
    app.include_router(analyses.router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": "PROJECT ORBIT API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
