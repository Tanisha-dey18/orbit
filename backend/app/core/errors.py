"""Domain errors and FastAPI exception handlers.

Error responses never include server filesystem paths or stack traces.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("orbit.errors")


class OrbitError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class ImageValidationError(OrbitError):
    status_code = 422
    code = "invalid_image"


class NotFoundError(OrbitError):
    status_code = 404
    code = "not_found"


class ModelUnavailableError(OrbitError):
    status_code = 409
    code = "model_unavailable"


class PayloadTooLargeError(OrbitError):
    status_code = 413
    code = "payload_too_large"


def _body(code: str, message: str, details=None) -> dict:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(OrbitError)
    async def _orbit_error(_: Request, exc: OrbitError):
        return JSONResponse(status_code=exc.status_code, content=_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [
            {"loc": [str(p) for p in err.get("loc", [])], "msg": err.get("msg", "")}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content=_body("validation_error", "Request validation failed.", details))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled server error: %s", type(exc).__name__)
        return JSONResponse(status_code=500, content=_body("internal_error", "An internal error occurred. Check server logs."))
