import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import HTTPException

from weather_plus.config import BREAKDOWN_DEG, SUPPORTED


def tile_id(lat: float, lon: float) -> str:
    def bin1(x, mn, st):
        idx = int(math.floor((x - mn) / st))
        return max(0, idx)

    i = bin1(lat, -90.0, BREAKDOWN_DEG)
    j = bin1(((lon + 180.0) % 360.0) - 180.0, -180.0, BREAKDOWN_DEG)
    return f"LAT{i}_LON{j}"


def times_between(start_iso: str, end_iso: str) -> List[datetime]:
    t0 = datetime.fromisoformat(start_iso)
    t1 = datetime.fromisoformat(end_iso)
    if t0 >= t1:
        raise HTTPException(400, "Empty time range")

    out = []
    t = t0
    while t < t1:
        out.append(t.replace(microsecond=0))
        t += timedelta(hours=1)
    return out


def extract_hourly(block: Dict, key: str) -> Optional[np.ndarray]:
    if not block:
        return None
    H = block.get("hourly", {})
    if key not in H:
        return None
    return np.asarray(H[key], dtype=float)


# ---------- small helpers for optional extras ----------


def _safe(arr, fill=0.0):
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1:
        a = a.ravel()
    return a


def _shift(a: np.ndarray, k: int):
    """Positive k = lag (shift right)."""
    if a is None:
        return None
    out = np.empty_like(a, dtype=float)
    if k <= 0:
        # lead or no-op
        if k == 0:
            return a.copy()
        out[:k] = a[-k:]
        out[k:] = a[-1]
        return out
    # lag
    out[:k] = a[0]
    out[k:] = a[:-k]
    return out


def _roll_mean(a: np.ndarray, win: int):
    if a is None:
        return None
    if win <= 1:
        return a.copy()
    # trailing window mean with left-edge padding
    pad = np.repeat(a[:1], win - 1)
    b = np.concatenate([pad, a])
    c = np.convolve(b, np.ones(win, dtype=float) / float(win), mode="valid")
    return c


def _deg2rad(x):  # robust
    return np.deg2rad(np.asarray(x, dtype=float))


def _to_uv_from_speed_dir(speed, deg):
    if (speed is None) or (deg is None):
        return None, None
    s = np.asarray(speed, dtype=float)
    th = _deg2rad(deg)
    # meteorological to math: wind coming from direction -> components blowing toward
    # Using u = -s * sin(dir), v = -s * cos(dir)
    u = -s * np.sin(th)
    v = -s * np.cos(th)
    return u, v


def assemble_X(
    bundle,
    var: str,
    lat: float,
    lon: float,
    times: List[datetime],
    base_om: Dict[str, np.ndarray],
    base_ifs: Optional[Dict[str, np.ndarray]],
):
    """
    Build the feature matrix expected by the saved bundle.

    - If bundle.feature_names is None: return legacy 6-col layout.
    - Otherwise: produce exactly the names listed in bundle.feature_names.
      This path now supports:
        baseline_om, baseline_ifs
        hod_sin, hod_cos, doy_sin, doy_cos
        tminus_td
        u100_om, v100_om
        dspfc_3h
        t2m_mean3, t2m_grad3
        skin_temp, snow_depth
        swvl1..swvl4  (from volumetric_soil_water_layer_1..4)
        lag1, lag3, lag6, lag24 (lags of baseline_om for this var)
        and {k}_om / {k}_ifs for any k in SUPPORTED, if present in baselines
    """
    T = len(times)
    hod = np.array([t.hour + t.minute / 60.0 for t in times], dtype=float)
    lead = np.arange(T, dtype=float)
    latv = np.full(T, float(lat))
    lonv = np.full(T, float(lon))

    feat_names = getattr(bundle, "feature_names", None)

    # ---------- legacy straight-6 ----------
    if feat_names is None:
        bom = base_om.get(var)
        if bom is None:
            raise HTTPException(500, f"Missing baseline OM {var}")
        bif = (base_ifs or {}).get(var, bom)
        return np.column_stack([latv, lonv, hod, lead, bom, bif]), [
            "lat",
            "lon",
            "hod",
            "lead",
            "baseline_om",
            "baseline_ifs",
        ]

    # ---------- rich feature builder ----------
    coll: Dict[str, np.ndarray] = {
        "lat": latv,
        "lon": lonv,
        "hod": hod,
        "lead": lead,
        "hod_sin": np.sin(2 * np.pi * hod / 24.0),
        "hod_cos": np.cos(2 * np.pi * hod / 24.0),
        "doy_sin": np.sin(
            2 * np.pi * np.array([t.timetuple().tm_yday for t in times]) / 365.25
        ),
        "doy_cos": np.cos(
            2 * np.pi * np.array([t.timetuple().tm_yday for t in times]) / 365.25
        ),
    }

    # Generic baselines for the *target variable*
    bom = _safe(base_om.get(var))
    if bom is None:
        raise HTTPException(500, f"Missing baseline OM {var}")
    bif = _safe((base_ifs or {}).get(var, bom))
    coll["baseline_om"] = bom
    coll["baseline_ifs"] = bif

    # Also expose per-variable *_om / *_ifs for any SUPPORTED baseline present
    # (models may have selected some of these)
    for k in SUPPORTED:
        v_om = _safe(base_om.get(k))
        if v_om is not None:
            coll[f"{k}_om"] = v_om
        if base_ifs:
            v_if = _safe(base_ifs.get(k))
            if v_if is not None:
                coll[f"{k}_ifs"] = v_if

    # Derived: temperature minus dew point (needs OM)
    if ("temperature_2m" in base_om) and ("dew_point_2m" in base_om):
        t2m = _safe(base_om["temperature_2m"])
        td2m = _safe(base_om["dew_point_2m"])
        if (t2m is not None) and (td2m is not None):
            coll["tminus_td"] = t2m - td2m

    # Derived wind components from 100m wind (OM)
    if ("wind_speed_100m" in base_om) and ("wind_direction_100m" in base_om):
        u, v = _to_uv_from_speed_dir(
            base_om["wind_speed_100m"], base_om["wind_direction_100m"]
        )
        if (u is not None) and (v is not None):
            coll["u100_om"] = u
            coll["v100_om"] = v

    # Optional extras used by training PREF (compute only if requested later):
    # - Surface pressure 3h change
    if "surface_pressure" in base_om:
        ps = _safe(base_om["surface_pressure"])
        if ps is not None:
            coll["dspfc_3h"] = ps - _shift(ps, 3)

    # - Temp trailing 3h mean and 3h gradient
    if "temperature_2m" in base_om:
        t2 = _safe(base_om["temperature_2m"])
        if t2 is not None:
            coll["t2m_mean3"] = _roll_mean(t2, 3)
            coll["t2m_grad3"] = t2 - _shift(t2, 3)

    # - Skin temperature, snow depth (pass-through if present)
    if "soil_temperature_0cm" in base_om:
        coll["skin_temp"] = _safe(base_om["soil_temperature_0cm"])
    if "snow_depth" in base_om:
        coll["snow_depth"] = _safe(base_om["snow_depth"])

    if "soil_moisture_0_to_1cm" in base_om:
        coll["soil_moisture_0_to_1cm"] = _safe(base_om["soil_moisture_0_to_1cm"])
    if "soil_moisture_1_to_3cm" in base_om:
        coll["soil_moisture_1_to_3cm"] = _safe(base_om["soil_moisture_1_to_3cm"])
    if "soil_moisture_3_to_9cm" in base_om:
        coll["soil_moisture_3_to_9cm"] = _safe(base_om["soil_moisture_3_to_9cm"])
    if "soil_moisture_9_to_27cm" in base_om:
        coll["soil_moisture_9_to_27cm"] = _safe(base_om["soil_moisture_9_to_27cm"])
    if "soil_moisture_27_to_81cm" in base_om:
        coll["soil_moisture_27_to_81cm"] = _safe(base_om["soil_moisture_27_to_81cm"])

    have_soil_moisture = all(
        k in coll and coll[k] is not None
        for k in [
            "soil_moisture_0_to_1cm",
            "soil_moisture_1_to_3cm",
            "soil_moisture_3_to_9cm",
            "soil_moisture_9_to_27cm",
            "soil_moisture_27_to_81cm",
        ]
    )

    if have_soil_moisture:
        t = {
            "0_1": 1.0,
            "1_3": 2.0,
            "3_9": 6.0,
            "9_27": 18.0,
            "27_81": 54.0,
        }  # cm thickness
        sm = {
            "0_1": coll["soil_moisture_0_to_1cm"],
            "1_3": coll["soil_moisture_1_to_3cm"],
            "3_9": coll["soil_moisture_3_to_9cm"],
            "9_27": coll["soil_moisture_9_to_27cm"],
            "27_81": coll["soil_moisture_27_to_81cm"],
        }

        # swvl1 (~0–7 cm) ≈ weighted mean of 0–1, 1–3, 3–9
        num = t["0_1"] * sm["0_1"] + t["1_3"] * sm["1_3"] + t["3_9"] * sm["3_9"]
        den = t["0_1"] + t["1_3"] + t["3_9"]
        coll["swvl1"] = num / den

        # swvl2 (~7–28 cm) ≈ 9–27 plus a 7/54 slice of 27–81
        num = t["9_27"] * sm["9_27"] + (7.0 / 54.0) * t["27_81"] * sm["27_81"]
        den = t["9_27"] + (7.0 / 54.0) * t["27_81"]
        coll["swvl2"] = num / den

        # swvl3 (~28–100 cm) ≈ use the remaining 21/54 of 27–81 as a coarse proxy
        weight = (21.0 / 54.0) * t["27_81"]
        coll["swvl3"] = (weight * sm["27_81"]) / weight

        # swvl4 (100–289 cm) cannot be synthesized from 27–81; omit on purpose.
        coll["swvl4"] = np.full(T, np.nan, dtype=float)
    else:
        coll["swvl1"] = np.full(T, np.nan, dtype=float)
        coll["swvl2"] = np.full(T, np.nan, dtype=float)
        coll["swvl3"] = np.full(T, np.nan, dtype=float)
        coll["swvl4"] = np.full(T, np.nan, dtype=float)

    # - Lag features of the *target variable's* baseline_om
    #   (lag1, lag3, lag6, lag24). Only compute when requested.
    lag_cache = {
        1: _shift(bom, 1),
        3: _shift(bom, 3),
        6: _shift(bom, 6),
        24: _shift(bom, 24),
    }
    coll["lag1"] = lag_cache[1]
    coll["lag3"] = lag_cache[3]
    coll["lag6"] = lag_cache[6]
    coll["lag24"] = lag_cache[24]

    # Now, assemble exactly the requested columns:
    Xcols = []
    data_cols = []
    for name in getattr(bundle, "feature_names", []):
        if name not in coll:
            raise HTTPException(500, f"Server cannot build feature '{name}'.")
        Xcols.append(name)
        data_cols.append(coll[name])
        
    print(Xcols, data_cols)

    X = np.column_stack(data_cols) if Xcols else np.zeros((T, 0), dtype=float)
    return X, Xcols
