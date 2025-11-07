from __future__ import annotations
import time
from typing import Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException

from ..config import MODEL_DIR
from .schemas import ForecastRequest, ForecastResponse
from ..engine.feature_store import get_baselines, expand_grid, make_basic_features
from ..engine.calibration import Calibrator

app = FastAPI(title="Weather+ Forecast API", version="0.2.0")
cal = Calibrator(model_dir=MODEL_DIR)

OM_UNITS = {
    "time": "iso8601",
    "temperature_2m": "°C",
    "dew_point_2m": "°C",
    "surface_pressure": "hPa",
    "precipitation": "mm",
    "wind_speed_100m": "km/h",
    "wind_direction_100m": "°",
}


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    t0 = time.time()
    hourly = req.hourly if isinstance(req.hourly, list) else [req.hourly]

    # 1) pull **both** baselines (OM blend + ECMWF IFS HRES)
    try:
        bases = get_baselines(
            req.latitude,
            req.longitude,
            hourly,
            req.start_hour,
            req.end_hour,
            req.timezone,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Baseline provider failed: {e}")

    base_om = bases["om"]
    hourly_data_om = base_om.get("hourly", {})
    times = hourly_data_om.get("time")
    if times is None:
        raise HTTPException(status_code=500, detail="OM baseline missing hourly.time")

    grid = expand_grid(req.latitude, req.longitude)
    G = len(grid)
    T = len(times)

    # Optional second baseline
    base_ifs = bases.get("ifs")
    hourly_data_ifs = base_ifs.get("hourly", {}) if base_ifs else {}

    # 2) Features
    X_basic = make_basic_features(grid, times)  # [T*G, 4]

    out_hourly: Dict[str, Any] = {"time": times}
    for name in hourly:
        if name == "time":
            continue
        # OM baseline
        arr_om = np.array(hourly_data_om.get(name))
        if arr_om.ndim == 1:
            arr_om = arr_om[:, None]
        if arr_om.shape != (T, G):
            if arr_om.size == T * G:
                arr_om = arr_om.reshape(T, G)
            else:
                raise HTTPException(status_code=500, detail=f"OM {name} wrong shape")
        vec_om = arr_om.reshape(-1)

        # IFS baseline (optional)
        if hourly_data_ifs and name in hourly_data_ifs:
            arr_ifs = np.array(hourly_data_ifs[name])
            if arr_ifs.ndim == 1:
                arr_ifs = arr_ifs[:, None]
            if arr_ifs.shape != (T, G):
                if arr_ifs.size == T * G:
                    arr_ifs = arr_ifs.reshape(T, G)
                else:
                    arr_ifs = None
            vec_ifs = (
                arr_ifs.reshape(-1)
                if arr_ifs is not None
                else np.full_like(vec_om, np.nan)
            )
        else:
            vec_ifs = np.full_like(vec_om, np.nan)

        # Full feature vector: [lat, lon, hod, lead, baseline_om, baseline_ifs, baseline_diff]
        base_diff = np.where(np.isfinite(vec_ifs), vec_om - vec_ifs, 0.0)
        X = np.column_stack(
            [X_basic, vec_om, np.nan_to_num(vec_ifs, nan=vec_om), base_diff]
        )

        yhat = cal.predict(name, X, baseline=vec_om)

        # Physical clamping in OM units
        if name in ("temperature_2m", "dew_point_2m"):
            yhat = np.clip(yhat, -90.0, 60.0)
        elif name == "surface_pressure":
            yhat = np.clip(yhat, 800.0, 1100.0)
        elif name == "precipitation":
            yhat = np.clip(yhat, 0.0, None)
        elif name == "wind_speed_100m":
            yhat = np.clip(yhat, 0.0, None)
        elif name == "wind_direction_100m":
            yhat = np.mod(yhat, 360.0)

        out_hourly[name] = yhat.reshape(T, G).tolist()

    resp = {
        "latitude": req.latitude,
        "longitude": req.longitude,
        "generationtime_ms": (time.time() - t0) * 1000.0,
        "utc_offset_seconds": 0,
        "timezone": req.timezone or "UTC",
        "timezone_abbreviation": "UTC",
        "hourly_units": {k: OM_UNITS.get(k, "unknown") for k in ["time", *hourly]},
        "hourly": out_hourly,
    }
    return resp
