#!/usr/bin/env python
import os, json, argparse, datetime as dt
import requests
from urllib.parse import urlencode

OM_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
OUT_DIR = os.path.join("data", "om_baseline")

DEFAULT_HOURLY = [
    "temperature_2m",
    "precipitation",
    "wind_speed_100m",
    "wind_direction_100m",
    "dew_point_2m",
    "surface_pressure",
]


def daterange(start: dt.datetime, end: dt.datetime, chunk_hours=168):
    t = start
    while t < end:
        t2 = min(t + dt.timedelta(hours=chunk_hours), end)
        yield t, t2
        t = t2


def make_retry_session():
    # Optional: resilient HTTP client with retries & timeouts
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
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        return s
    except Exception:
        return requests.Session()


def get_om(
    session, lat_list, lon_list, hourly, start_hour, end_hour, models=None, timeout=120
):
    # Build params per Open-Meteo docs:
    # - GET with query params
    # - multiple coords allowed as comma-separated lists
    # - time slicing via start_hour / end_hour (ISO8601)
    # - model selection via "models" (plural)
    params = {
        "latitude": ",".join(f"{x:.6f}" for x in lat_list),
        "longitude": ",".join(f"{x:.6f}" for x in lon_list),
        "hourly": ",".join(hourly),
        "start_hour": start_hour,  # e.g. 2025-11-01T00:00
        "end_hour": end_hour,  # e.g. 2025-11-07T00:00
        "timezone": "UTC",
        "timeformat": "iso8601",
    }
    if models:
        params["models"] = models  # e.g. "ecmwf_ifs"
    r = session.get(OM_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--lat-steps", type=int, default=5)
    ap.add_argument("--lon-steps", type=int, default=5)
    ap.add_argument("--start", type=str, required=True)  # ISO like 2025-11-01T00:00
    ap.add_argument("--end", type=str, required=True)  # ISO
    ap.add_argument("--hourly", nargs="+", default=DEFAULT_HOURLY)
    ap.add_argument("--chunk-hours", type=int, default=168)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    lat_list = [
        args.lat_min + i * (args.lat_max - args.lat_min) / max(args.lat_steps - 1, 1)
        for i in range(args.lat_steps)
    ]
    lon_list = [
        args.lon_min + j * (args.lon_max - args.lon_min) / max(args.lon_steps - 1, 1)
        for j in range(args.lon_steps)
    ]

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    session = make_retry_session()

    for a, b in daterange(t0, t1, args.chunk_hours):
        s = a.strftime("%Y-%m-%dT%H:%M")
        e = b.strftime("%Y-%m-%dT%H:%M")

        # Best-match (combined models)
        om = get_om(session, lat_list, lon_list, args.hourly, s, e, models=None)

        # ECMWF IFS HRES explicitly
        ifs = None
        try:
            ifs = get_om(
                session, lat_list, lon_list, args.hourly, s, e, models="ecmwf_ifs"
            )
        except Exception as ex:
            # Non-fatal: keep proceeding even if ECMWF request fails
            print(f"[warn] ecmwf_ifs request failed for {s}..{e}: {ex}")

        out = {
            "meta": {
                "lat": lat_list,
                "lon": lon_list,
                "start": s,
                "end": e,
                "hourly": args.hourly,
                "url_base": OM_URL,
            },
            "om": om,
            "ifs": ifs,
        }

        name = f"omifs_{a.strftime('%Y%m%d%H')}_{b.strftime('%Y%m%d%H')}.json"
        with open(os.path.join(OUT_DIR, name), "w") as f:
            json.dump(out, f)
        print("saved", name)


if __name__ == "__main__":
    main()
