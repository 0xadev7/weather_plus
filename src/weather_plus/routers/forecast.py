import logging
import time
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
    # Correlate with middleware request id if present
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

    # Utility to convert OM block -> {var: np.array([...])}
    def _blk_to_map(block):
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
            bundle = MC.load(var, tid)
            t1 = time.perf_counter()

            # Back-compat: simple estimators may have no feature_names/meta
            if not hasattr(bundle, "feature_names"):
                legacy = bundle
                bom = base_om.get(var)
                if bom is None:
                    log.error(
                        "req=%s pt=%d var=%s legacy missing baseline OM", req_id, i, var
                    )
                    raise HTTPException(500, f"Missing legacy baseline {var}")
                bif = base_ifs.get(var, bom)
                hod = np.array([t.hour + t.minute / 60.0 for t in times], dtype=float)
                lead = np.arange(len(times), dtype=float)
                X = np.column_stack(
                    [np.full_like(hod, la), np.full_like(hod, lo), hod, lead, bom, bif]
                )
                y = legacy.predict(X)
                H[var] = np.asarray(y, dtype=float).tolist()
                t2 = time.perf_counter()
                log.info(
                    "req=%s var=%s tile=%s task=legacy n=%d load=%.3fs infer=%.3fs",
                    req_id,
                    var,
                    tid,
                    len(times),
                    (t1 - t0),
                    (t2 - t1),
                )
                continue

            # Modern bundles
            bundle.meta = getattr(bundle, "meta", {"task": "reg"})
            X, _ = assemble_X(bundle, var, la, lo, times, base_om, base_ifs)
            task = bundle.meta.get("task", "reg")

            # Predict according to task kind
            if task == "tp2stage":
                clf, reg = bundle.model
                p = np.clip(clf.predict_proba(X)[:, 1], 0, 1)
                y = np.expm1(reg.predict(X))
                out_vals = (p * y).astype(float)
            elif task == "wspd":
                y = bundle.model.predict(X)
                out_vals = np.clip(y**2, 0, None).astype(float)
            elif task == "wdir":
                ms, mc = bundle.model
                s = ms.predict(X)
                c = mc.predict(X)
                ang = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
                out_vals = ang.astype(float)
            else:
                out_vals = bundle.model.predict(X).astype(float)

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
            if log.isEnabledFor(logging.DEBUG):
                # quick sanity stats
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
