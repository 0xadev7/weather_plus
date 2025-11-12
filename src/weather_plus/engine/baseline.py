import requests
from fastapi import HTTPException

from weather_plus.config import OPEN_METEO_URL, OM_TIMEOUT, BASELINE_NEEDED


def fetch_openmeteo(
    latitude_csv: str, longitude_csv: str, start: str, end: str, models=None
):
    params = {
        "latitude": latitude_csv,
        "longitude": longitude_csv,
        "hourly": BASELINE_NEEDED,
        "start_hour": start,
        "end_hour": end,
        "timezone": "UTC",
        "timeformat": "iso8601",
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
