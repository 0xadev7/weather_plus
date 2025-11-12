import logging
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
from fastapi import APIRouter, Query, HTTPException, Request

from weather_plus.engine import baseline
from weather_plus.config import SUPPORTED, USE_BASELINE
from weather_plus.engine.model_cache import ModelCache
from weather_plus.engine.features import (
    tile_id,
    times_between,
    extract_hourly,
    assemble_X,
)

router = APIRouter()
log = logging.getLogger("weather_plus.api")

# Lazy import of config for MODEL_DIR to keep start time fast & avoid circulars
MC = ModelCache(
    model_dir=__import__("src.weather_plus.config", fromlist=[""]).MODEL_DIR
)  # noqa: E402


def _mk_assemble_adapter(feature_names):
    """assemble_X expects an object with .feature_names."""
    return SimpleNamespace(feature_names=feature_names or [])


@router.get("/v1/forecast")
def forecast(
    request: Request,
    latitude: str = Query(..., description="Comma-separated"),
    longitude: str = Query(..., description="Comma-separated"),
    hourly: str = Query(..., description="Comma-separated hourly var names"),
    start_hour: str = Query(...),
    end_hour: str = Query(...),
    timezone: str = Query("UTC"),
    timeformat: str = Query("iso8601"),
    models: Optional[str] = Query(None),
):
    req_id = request.headers.get("x-request-id", "n/a")

    # ---------- Parse inputs ----------
    try:
        lats = [float(x) for x in latitude.split(",") if x.strip()]
        lons = [float(x) for x in longitude.split(",") if x.strip()]
    except ValueError as e:
        log.warning("req=%s bad lat/lon parse: %s", req_id, e)
        raise HTTPException(400, "latitude/longitude must be comma-separated floats")

    if len(lats) != len(lons):
        log.warning(
            "req=%s mismatch lat_count=%d lon_count=%d", req_id, len(lats), len(lons)
        )
        raise HTTPException(400, "latitude and longitude lengths differ")

    vars_req = [v.strip() for v in hourly.split(",") if v.strip()]
    bad = [v for v in vars_req if v not in SUPPORTED]
    if bad:
        log.warning("req=%s unsupported vars=%s", req_id, bad)
        raise HTTPException(400, f"Unsupported variable(s): {', '.join(bad)}")

    times = times_between(start_hour, end_hour)
    log.info(
        "req=%s grid=%d points vars=%s window=[%s .. %s] timezone=%s",
        req_id,
        len(lats),
        ",".join(vars_req),
        start_hour,
        end_hour,
        timezone,
    )
    log.debug(
        "req=%s lats[0:5]=%s lons[0:5]=%s n_times=%d",
        req_id,
        lats[:5],
        lons[:5],
        len(times),
    )

    # ---------- Fetch baselines (optional) ----------
    om_points = ifs_points = None
    if USE_BASELINE:
        t0 = time.perf_counter()
        om_points = baseline.fetch_openmeteo(
            latitude, longitude, start_hour, end_hour, models=None
        )
        t1 = time.perf_counter()
        ifs_ok = True
        try:
            ifs_points = baseline.fetch_openmeteo(
                latitude, longitude, start_hour, end_hour, models="ecmwf_ifs025"
            )
        except Exception as e:
            ifs_ok = False
            ifs_points = None
            log.warning("req=%s ifs fetch failed: %s", req_id, e)
        t2 = time.perf_counter()

        if not isinstance(om_points, list) or len(om_points) != len(lats):
            log.error(
                "req=%s baseline mismatch om_points=%s len=%s expected=%s",
                req_id,
                type(om_points).__name__,
                getattr(om_points, "__len__", lambda: "?")(),
                len(lats),
            )
            raise HTTPException(502, "Baseline response didn't match requested grid")

        log.info(
            "req=%s baseline: openmeteo t=%.3fs points=%d; ifs t=%.3fs ok=%s",
            req_id,
            (t1 - t0),
            len(om_points),
            (t2 - t1),
            ifs_ok,
        )

    def _blk_to_map(block):
        # {var: np.array([...])} for all SUPPORTED, if present
        return {k: extract_hourly(block, k) for k in SUPPORTED}

    out = []
    # ---------- Per-point inference ----------
    for i, (la, lo) in enumerate(zip(lats, lons)):
        tid = tile_id(la, lo)
        H = {"time": [t.replace(microsecond=0).isoformat() for t in times]}
        om_pt = om_points[i] if om_points else None
        ifs_pt = ifs_points[i] if ifs_points else None
        base_om = _blk_to_map(om_pt) if om_pt else {}
        base_ifs = _blk_to_map(ifs_pt) if ifs_pt else {}

        log.debug("req=%s pt=%d lat=%.4f lon=%.4f tile=%s", req_id, i, la, lo, tid)

        for var in vars_req:
            t0 = time.perf_counter()
            bundle_obj = MC.load(var, tid)  # dict with model, feature_names, meta.task
            t1 = time.perf_counter()

            model = bundle_obj.get("model")
            feature_names = bundle_obj.get("feature_names", [])
            task = (bundle_obj.get("meta") or {}).get("task", "reg")

            adapter = _mk_assemble_adapter(feature_names)
            X, _ = assemble_X(adapter, var, la, lo, times, base_om, base_ifs)

            if task == "tp2stage":
                clf, reg = model
                p = np.clip(clf.predict_proba(X)[:, 1], 0, 1)
                y = np.expm1(reg.predict(X))
                out_vals = (p * y).astype(float)
            elif task == "wspd":
                y = model.predict(X)
                out_vals = np.clip(y**2, 0, None).astype(float)
            elif task == "wdir":
                ms, mc = model
                s = ms.predict(X)
                c = mc.predict(X)
                ang = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
                out_vals = ang.astype(float)
            else:
                out_vals = model.predict(X).astype(float)

            H[var] = out_vals.tolist()
            t2 = time.perf_counter()
            log.info(
                "req=%s var=%s tile=%s task=%s n=%d load=%.3fs infer=%.3fs",
                req_id,
                var,
                tid,
                task,
                len(out_vals),
                (t1 - t0),
                (t2 - t1),
            )
            if log.isEnabledFor(logging.DEBUG) and len(out_vals):
                log.debug(
                    "req=%s var=%s tile=%s stats min=%.4f max=%.4f mean=%.4f",
                    req_id,
                    var,
                    tid,
                    float(np.min(out_vals)),
                    float(np.max(out_vals)),
                    float(np.mean(out_vals)),
                )

        out.append({"latitude": la, "longitude": lo, "hourly": H})

    return out
