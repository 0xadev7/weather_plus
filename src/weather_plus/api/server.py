from __future__ import annotations
import time
from typing import Dict, Any, List
import numpy as np
from fastapi import FastAPI, HTTPException

from weather_plus.config import MODEL_DIR
from weather_plus.api.schemas import ForecastRequest, ForecastResponse
from weather_plus.engine.feature_store import (
    get_baselines,
    expand_grid,
    make_basic_features,
)
from weather_plus.engine.calibration import Calibrator
from weather_plus.engine.tiles import tile_id

app = FastAPI(title="Weather+ Forecast API", version="0.3.0")

# Loads per-variable artifacts; supports per-tile (var__TILE.joblib) and global (var.joblib)
cal = Calibrator(model_dir=MODEL_DIR)

# Units matching Open-Meteo
OM_UNITS = {
    "time": "iso8601",
    "temperature_2m": "°C",
    "dew_point_2m": "°C",
    "surface_pressure": "hPa",
    "precipitation": "mm",
    "wind_speed_100m": "km/h",
    "wind_direction_100m": "°",
}


def _ensure_tg(arr: np.ndarray, T: int, G: int, name: str) -> np.ndarray:
    """Ensure array is shaped [T, G]."""
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape == (T, G):
        return arr
    if arr.size == T * G:
        return arr.reshape(T, G)
    raise HTTPException(
        status_code=500,
        detail=f"Baseline {name} wrong shape {arr.shape}, expected {(T, G)}",
    )


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    t0 = time.time()
    hourly: List[str] = req.hourly if isinstance(req.hourly, list) else [req.hourly]

    # 1) Pull both baselines (Open-Meteo default blend + ECMWF IFS HRES via OM)
    try:
        bases = get_baselines(
            latitude=req.latitude,
            longitude=req.longitude,
            hourly=hourly,
            start_hour=req.start_hour,
            end_hour=req.end_hour,
            timezone=req.timezone,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Baseline provider failed: {e}")

    base_om = bases["om"]
    hourly_om = base_om.get("hourly", {})
    times = hourly_om.get("time")
    if times is None:
        raise HTTPException(status_code=500, detail="OM baseline missing hourly.time")

    base_ifs = bases.get("ifs")
    hourly_ifs = base_ifs.get("hourly", {}) if base_ifs else {}

    # 2) Grid and features
    grid = expand_grid(req.latitude, req.longitude)  # [(lat,lon), ...] length G
    G = len(grid)
    T = len(times)
    X_basic = make_basic_features(grid, times)  # [T*G, 4] => [lat, lon, hod, lead]

    # Precompute tile id per grid point, then repeat across time for vectorized routing
    tile_per_g = [tile_id(la, lo) for (la, lo) in grid]
    tiles_vec = np.concatenate(
        [np.array(tile_per_g) for _ in range(T)], axis=0
    )  # [T*G]

    # 3) Build outputs (OM-compatible)
    out_hourly: Dict[str, Any] = {"time": times}

    for name in hourly:
        if name == "time":
            continue

        # Baseline arrays in OM units
        arr_om = np.array(hourly_om.get(name))
        arr_om = _ensure_tg(arr_om, T, G, name)
        vec_om = arr_om.reshape(-1)  # [T*G]

        if name in hourly_ifs:
            arr_ifs = np.array(hourly_ifs[name])
            try:
                arr_ifs = _ensure_tg(arr_ifs, T, G, f"IFS:{name}")
                vec_ifs = arr_ifs.reshape(-1)
            except HTTPException:
                vec_ifs = np.full_like(vec_om, np.nan)
        else:
            vec_ifs = np.full_like(vec_om, np.nan)

        # Features: [lat, lon, hod, lead, baseline_om, baseline_ifs_filled, baseline_diff]
        vec_ifs_filled = np.nan_to_num(vec_ifs, nan=vec_om)
        base_diff = vec_om - vec_ifs_filled
        X = np.column_stack([X_basic, vec_om, vec_ifs_filled, base_diff])  # [T*G, 7]

        # 4) Tile-aware calibration (batch by tile for fewer Python calls)
        yhat = np.empty_like(vec_om, dtype=float)
        for tile in sorted(set(tiles_vec)):
            sel = tiles_vec == tile
            yhat_tile = cal.predict(var=name, X=X[sel], baseline=vec_om[sel], tile=tile)
            yhat[sel] = yhat_tile

        # 5) Physical clamps (keep OM units)
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
