import requests
from fastapi import HTTPException

from weather_plus.config import OPEN_METEO_URL, OM_TIMEOUT, BASELINE_NEEDED


def fetch_openmeteo(lat_seq, lon_seq, start_hour, end_hour, models=None):
    params = {
        "latitude": ",".join(f"{x:.6f}" for x in lat_seq),
        "longitude": ",".join(f"{x:.6f}" for x in lon_seq),
        "hourly": BASELINE_NEEDED,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "timezone": "UTC",
        "timeformat": "iso8601",
        "cell_selection": "nearest",
    }

    if models:
        params["models"] = models

    r = requests.get(OPEN_METEO_URL, params=params, timeout=OM_TIMEOUT)
    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Baseline fetch failed ({r.status_code})"
        )
    j = r.json()

    return j if isinstance(j, list) else [j]
