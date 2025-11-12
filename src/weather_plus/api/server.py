import logging
import os
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from weather_plus.routers.forecast import router as forecast_router

# ---------- Logging setup ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("weather_plus.server")

app = FastAPI(title="OpenMeteo-Compatible ML Forecast")
app.include_router(forecast_router)


# ---------- Per-request logging middleware ----------
@app.middleware("http")
async def add_logging(request: Request, call_next: Callable):
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    path = request.url.path
    q = request.url.query
    client = request.client.host if request.client else "unknown"

    log.info(
        "req=%s START %s %s%s client=%s",
        req_id,
        request.method,
        path,
        f"?{q}" if q else "",
        client,
    )
    try:
        response = await call_next(request)
    except Exception as e:
        # Log exception with traceback and return 500
        log.exception("req=%s EXC %s: %s", req_id, type(e).__name__, str(e))
        elapsed = time.perf_counter() - start
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "request_id": req_id,
                "detail": str(e),
                "elapsed_s": round(elapsed, 3),
            },
        )

    elapsed = time.perf_counter() - start
    size = response.headers.get("content-length", "?")
    log.info(
        "req=%s END %s t=%.3fs bytes=%s", req_id, response.status_code, elapsed, size
    )
    # propagate request id so callers can correlate
    response.headers["x-request-id"] = req_id
    response.headers["x-runtime-s"] = f"{elapsed:.3f}"
    return response


@app.get("/health")
def health_check():
    return {"status": "ok"}
