from __future__ import annotations
import os, time, datetime as dt
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import requests

from weather_plus.config import OPEN_METEO_URL, BASELINE_PROVIDER


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _post_om(
    hourly,
    latitude,
    longitude,
    start_hour,
    end_hour,
    timezone,
    model: Optional[str] = None,
):
    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": hourly,
        "start_hour": start_hour,
        "end_hour": end_hour,
    }
    if timezone:
        payload["timezone"] = timezone
    if model is not None:
        payload["model"] = (
            model  # Open-Meteo supports model filtering (validator uses "ecmwf_ifs")
        )
    r = requests.post(OPEN_METEO_URL, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def get_baselines(
    latitude, longitude, hourly, start_hour, end_hour, timezone
) -> Dict[str, Any]:
    # Always pull OM default blend
    om = _post_om(
        hourly, latitude, longitude, start_hour, end_hour, timezone, model=None
    )
    # And ECMWF IFS HRES (validator logs this baseline too)
    try:
        ifs = _post_om(
            hourly,
            latitude,
            longitude,
            start_hour,
            end_hour,
            timezone,
            model="ecmwf_ifs",
        )
    except Exception:
        ifs = None
    return {"om": om, "ifs": ifs}


def expand_grid(lat: List[float], lon: List[float]) -> List[Tuple[float, float]]:
    return [(la, lo) for la in lat for lo in lon]


def make_basic_features(
    grid: List[Tuple[float, float]], times_iso: List[str]
) -> np.ndarray:
    """[lat, lon, hour_of_day, lead_hours] for each (t,g) pair."""
    n_pts = len(grid)
    n_t = len(times_iso)
    lat = np.repeat([g[0] for g in grid], n_t)
    lon = np.repeat([g[1] for g in grid], n_t)
    t0 = dt.datetime.fromisoformat(times_iso[0].replace("Z", ""))
    leads = []
    hods = []
    for t in times_iso:
        tt = dt.datetime.fromisoformat(t.replace("Z", ""))
        leads.append((tt - t0).total_seconds() / 3600.0)
        hods.append(tt.hour + tt.minute / 60.0)
    leads = np.tile(np.array(leads), n_pts)
    hods = np.tile(np.array(hods), n_pts)
    return np.vstack([lat, lon, hods, leads]).T  # [n,4]
