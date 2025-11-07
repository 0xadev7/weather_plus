from __future__ import annotations
import os, time, datetime as dt
from typing import Dict, Any, List, Tuple, Optional, Sequence, Union
import numpy as np
import requests

from weather_plus.config import OPEN_METEO_URL

# ----------------------------
# Helpers
# ----------------------------


def _as_list(x: Union[float, Sequence[float]]) -> List[float]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [float(x)]


def _fmt_coord_list(vals: Sequence[float]) -> str:
    # Open-Meteo accepts comma-separated coordinate lists
    return ",".join(f"{v:.6f}" for v in vals)


def _make_retry_session() -> requests.Session:
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s
    except Exception:
        return requests.Session()


_SESSION = _make_retry_session()

# ----------------------------
# Open-Meteo fetch (GET, models=)
# ----------------------------


def _get_om(
    hourly: Sequence[str],
    latitude: Union[float, Sequence[float]],
    longitude: Union[float, Sequence[float]],
    start_hour: str,
    end_hour: str,
    timezone: Optional[str],
    model: Optional[str] = None,
    timeout: int = 90,
) -> Dict[str, Any]:
    """
    Calls Open-Meteo /v1/forecast using GET.
    - Multiple coords supported via comma-separated strings.
    - Model selection via `models` (plural), e.g., 'ecmwf_ifs'.
    """
    lat_list = _as_list(latitude)
    lon_list = _as_list(longitude)

    params = {
        "latitude": _fmt_coord_list(lat_list),
        "longitude": _fmt_coord_list(lon_list),
        "hourly": ",".join(hourly),
        "start_hour": start_hour,
        "end_hour": end_hour,
        "timeformat": "iso8601",
    }
    if timezone:
        params["timezone"] = timezone
    if model:
        params["models"] = model  # NOTE: 'models' (plural) is required by Open-Meteo

    r = _SESSION.get(OPEN_METEO_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# Backward-compatible alias (if anything else imports _post_om)
_post_om = _get_om

# ----------------------------
# Public API
# ----------------------------


def get_baselines(
    latitude: Union[float, Sequence[float]],
    longitude: Union[float, Sequence[float]],
    hourly: Sequence[str],
    start_hour: str,
    end_hour: str,
    timezone: Optional[str],
) -> Dict[str, Any]:
    """
    Returns:
      {
        "om":  <Open-Meteo best-match/blended>,
        "ifs": <ECMWF IFS HRES or None on failure>
      }
    """
    # Best-match / blended (no explicit model)
    om = _get_om(
        hourly, latitude, longitude, start_hour, end_hour, timezone, model=None
    )

    # ECMWF IFS HRES explicitly
    try:
        ifs = _get_om(
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
