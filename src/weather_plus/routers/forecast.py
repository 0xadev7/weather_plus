from fastapi import APIRouter, Query, HTTPException
import numpy as np
from typing import Optional, Dict, List

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
MC = ModelCache(
    model_dir=__import__("src.weather_plus.config", fromlist=[""]).MODEL_DIR
)  # lazy import cfg


@router.get("/v1/forecast")
def forecast(
    latitude: str = Query(..., description="Comma-separated"),
    longitude: str = Query(..., description="Comma-separated"),
    hourly: str = Query(..., description="Comma-separated hourly var names"),
    start_hour: str = Query(...),
    end_hour: str = Query(...),
    timezone: str = Query("UTC"),
    timeformat: str = Query("iso8601"),
    models: Optional[str] = Query(None),
):
    lats = [float(x) for x in latitude.split(",") if x.strip()]
    lons = [float(x) for x in longitude.split(",") if x.strip()]
    if len(lats) != len(lons):
        raise HTTPException(400, "latitude and longitude lengths differ")
    vars_req = [v.strip() for v in hourly.split(",") if v.strip()]
    for v in vars_req:
        if v not in SUPPORTED:
            raise HTTPException(400, f"Unsupported variable: {v}")

    times = times_between(start_hour, end_hour)

    om_points = ifs_points = None
    if USE_BASELINE:
        om_points = baseline.fetch_openmeteo(
            latitude, longitude, start_hour, end_hour, models=None
        )
        try:
            ifs_points = baseline.fetch_openmeteo(
                latitude, longitude, start_hour, end_hour, models="ecmwf_ifs025"
            )
        except Exception:
            ifs_points = None
        if not isinstance(om_points, list) or len(om_points) != len(lats):
            raise HTTPException(502, "Baseline response didn't match requested grid")

    out = []
    for i, (la, lo) in enumerate(zip(lats, lons)):
        H = {"time": [t.replace(microsecond=0).isoformat() for t in times]}
        om_pt = om_points[i] if om_points else None
        ifs_pt = ifs_points[i] if ifs_points else None

        # prepare numpy baseline maps
        def _blk_to_map(block):
            return {k: extract_hourly(block, k) for k in SUPPORTED}

        base_om = _blk_to_map(om_pt) if om_pt else {}
        base_ifs = _blk_to_map(ifs_pt) if ifs_pt else {}

        tid = tile_id(la, lo)
        for var in vars_req:
            bundle = MC.load(var, tid)
            # back-compat for simple ridge models (no feature_names)
            if not hasattr(bundle, "feature_names"):
                # some saved objects might be plain estimators/wrappers with .predict
                legacy = bundle
                bom = base_om.get(var)
                if bom is None:
                    raise HTTPException(500, f"Missing legacy baseline {var}")
                bif = base_ifs.get(var, bom)
                hod = np.array([t.hour + t.minute / 60.0 for t in times], dtype=float)
                lead = np.arange(len(times), dtype=float)
                X = np.column_stack(
                    [np.full_like(hod, la), np.full_like(hod, lo), hod, lead, bom, bif]
                )
                y = legacy.predict(X)
                H[var] = np.asarray(y, dtype=float).tolist()
            else:
                bundle.meta = getattr(bundle, "meta", {"task": "reg"})
                X, _ = assemble_X(bundle, var, la, lo, times, base_om, base_ifs)
                task = bundle.meta.get("task", "reg")
                if task == "tp2stage":
                    clf, reg = bundle.model
                    p = np.clip(clf.predict_proba(X)[:, 1], 0, 1)
                    y = np.expm1(reg.predict(X))
                    H[var] = (p * y).astype(float).tolist()
                elif task == "wspd":
                    y = bundle.model.predict(X)
                    H[var] = np.clip(y**2, 0, None).astype(float).tolist()
                elif task == "wdir":
                    ms, mc = bundle.model
                    s = ms.predict(X)
                    c = mc.predict(X)
                    ang = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
                    H[var] = ang.astype(float).tolist()
                else:
                    H[var] = bundle.model.predict(X).astype(float).tolist()
        out.append({"latitude": la, "longitude": lo, "hourly": H})
    return out
