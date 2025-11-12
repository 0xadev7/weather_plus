import requests
from fastapi import HTTPException

from weather_plus.config import OPEN_METEO_URL, OM_TIMEOUT, BASELINE_NEEDED


def make_retry_session():
    # We’ll handle 429 ourselves (Retry-After), let HTTPAdapter retry connections.
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=20))
        s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=20))
        return s
    except Exception:
        return requests.Session()


def fetch_openmeteo(lat_seq, lon_seq, start_hour, end_hour, models=None):

    session = make_retry_session()

    if len(lat_seq) != len(lon_seq):
        raise ValueError("latitude and longitude must have the same number of elements")

    params = {
        "latitude": ",".join(f"{x:.6f}" for x in lat_seq),
        "longitude": ",".join(f"{x:.6f}" for x in lon_seq),
        "hourly": ",".join(BASELINE_NEEDED),
        "start_hour": start_hour,
        "end_hour": end_hour,
        "timezone": "UTC",
        "timeformat": "iso8601",
        "cell_selection": "nearest",
    }
    if models:
        params["models"] = models

    r = session.get(OPEN_METEO_URL, params=params, timeout=OM_TIMEOUT)
    # Manually handle 429 to honor Retry-After in caller loop.
    if r.status_code == 429:
        return r  # caller inspects and sleeps
    r.raise_for_status()

    j = r.json()

    return j if isinstance(j, list) else [j]
