import json
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ServiceError(Exception):
    """Generic service-layer error."""


class DuplicateFileNameError(ServiceError):
    """Raised when a file name already exists in a vector store."""


def _map_exception_to_status_and_code(exc: Exception) -> tuple[int, str]:
    """Map known exceptions to (HTTP status, error_code)."""
    if isinstance(exc, DuplicateFileNameError):
        return 409, "DUPLICATE_FILE"
    if isinstance(exc, ServiceError):
        return 400, "SERVICE_ERROR"
    # Add more mappings as needed (e.g., NotFoundError -> 404)
    return 500, "INTERNAL_ERROR"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Attach correlation ID, log, and convert exceptions to JSON errors."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            status, code = _map_exception_to_status_and_code(exc)
            body = {
                "error_code": code,
                "message": str(exc),
                "correlation_id": correlation_id,
            }
            # Minimal structured logging (stdout); replace with loguru if preferred
            print(
                json.dumps(
                    {
                        "level": "ERROR",
                        "event": "request_failed",
                        "path": request.url.path,
                        "method": request.method,
                        "status": status,
                        "error_code": code,
                        "message": str(exc),
                        "correlation_id": correlation_id,
                    }
                )
            )
            return JSONResponse(status_code=status, content=body)

        # Add correlation ID header to successful responses
        response.headers["X-Correlation-ID"] = correlation_id

        # Basic access log
        duration_ms = (time.perf_counter() - start) * 1000
        print(
            json.dumps(
                {
                    "level": "INFO",
                    "event": "request_completed",
                    "path": request.url.path,
                    "method": request.method,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "correlation_id": correlation_id,
                }
            )
        )
        return response


def setup_middlewares(app: FastAPI) -> None:
    """Register the error-handling middleware."""
    app.add_middleware(ErrorHandlingMiddleware)
