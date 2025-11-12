import math, numpy as np
from datetime import datetime
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
        out.append(t)
        t = t.replace(microsecond=0) + (t1 - t0).__class__(hours=1)
    # fix increment: timedelta(hours=1)
    out = []
    t = t0
    from datetime import timedelta

    while t < t1:
        out.append(t)
        t += timedelta(hours=1)
    return out


def extract_hourly(block: Dict, key: str) -> Optional[np.ndarray]:
    if not block:
        return None
    H = block.get("hourly", {})
    if key not in H:
        return None
    return np.asarray(H[key], dtype=float)


def assemble_X(
    bundle,
    var: str,
    lat: float,
    lon: float,
    times: List[datetime],
    base_om: Dict[str, np.ndarray],
    base_ifs: Optional[Dict[str, np.ndarray]],
):
    T = len(times)
    hod = np.array([t.hour + t.minute / 60.0 for t in times], dtype=float)
    lead = np.arange(T, dtype=float)
    latv = np.full(T, float(lat))
    lonv = np.full(T, float(lon))

    feat_names = getattr(bundle, "feature_names", None)
    legacy_name = var

    if feat_names is None:
        bom = base_om.get(legacy_name)
        if bom is None:
            raise HTTPException(500, f"Missing baseline OM {legacy_name}")
        bif = (base_ifs or {}).get(legacy_name, bom)
        return np.column_stack([latv, lonv, hod, lead, bom, bif]), [
            "lat",
            "lon",
            "hod",
            "lead",
            "baseline_om",
            "baseline_ifs",
        ]

    # richer bundles
    coll = {
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
    for k in SUPPORTED:
        if k in base_om:
            coll[f"{k}_om"] = base_om[k]
        if base_ifs and k in base_ifs:
            coll[f"{k}_ifs"] = base_ifs[k]
    if (
        "tminus_td" in feat_names
        and ("temperature_2m" in base_om)
        and ("dew_point_2m" in base_om)
    ):
        coll["tminus_td"] = base_om["temperature_2m"] - base_om["dew_point_2m"]
    if ("wind_speed_100m" in base_om) and ("wind_direction_100m" in base_om):
        wspd = base_om["wind_speed_100m"]
        wdir = np.deg2rad(base_om["wind_direction_100m"])
        coll["u100_om"] = -wspd * np.sin(wdir)
        coll["v100_om"] = -wspd * np.cos(wdir)

    Xcols = []
    for name in feat_names:
        if name not in coll:
            raise HTTPException(500, f"Server cannot build feature '{name}'.")
        Xcols.append(name)
    X = np.column_stack([coll[n] for n in Xcols])
    return X, Xcols
