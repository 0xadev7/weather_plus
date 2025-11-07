#!/usr/bin/env python
import os, json, argparse, datetime as dt
import requests

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
    session, lat_seq, lon_seq, hourly, start_hour, end_hour, models=None, timeout=120
):
    """
    lat_seq and lon_seq must be equal-length lists of paired coordinates.
    """
    if len(lat_seq) != len(lon_seq):
        raise ValueError("latitude and longitude must have the same number of elements")

    params = {
        "latitude": ",".join(f"{x:.6f}" for x in lat_seq),
        "longitude": ",".join(f"{x:.6f}" for x in lon_seq),
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


def make_linspace(vmin: float, vmax: float, steps: int):
    if steps <= 1:
        return [vmin]
    return [vmin + i * (vmax - vmin) / (steps - 1) for i in range(steps)]


def make_grid_pairs(lat_list, lon_list):
    """
    Cartesian product of lat_list x lon_list, returned as two equal-length lists
    (latitudes and longitudes), so each index is a paired location.
    """
    lat_seq, lon_seq = [], []
    for lat in lat_list:
        for lon in lon_list:
            lat_seq.append(lat)
            lon_seq.append(lon)
    return lat_seq, lon_seq


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

    # Build 1D axes
    lat_list = make_linspace(args.lat_min, args.lat_max, args.lat_steps)
    lon_list = make_linspace(args.lon_min, args.lon_max, args.lon_steps)

    # Build paired grid (length = lat_steps * lon_steps)
    grid_lat, grid_lon = make_grid_pairs(lat_list, lon_list)
    assert len(grid_lat) == len(grid_lon), "Paired lat/lon must be same length"

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    session = make_retry_session()

    for a, b in daterange(t0, t1, args.chunk_hours):
        s = a.strftime("%Y-%m-%dT%H:%M")
        e = b.strftime("%Y-%m-%dT%H:%M")

        # Best-match (combined models)
        om = get_om(session, grid_lat, grid_lon, args.hourly, s, e, models=None)

        # ECMWF IFS HRES explicitly
        ifs = None
        try:
            ifs = get_om(
                session, grid_lat, grid_lon, args.hourly, s, e, models="ecmwf_ifs"
            )
        except Exception as ex:
            print(f"[warn] ecmwf_ifs request failed for {s}..{e}: {ex}")

        out = {
            "meta": {
                "lat_axis": lat_list,
                "lon_axis": lon_list,
                "paired_lat": grid_lat,
                "paired_lon": grid_lon,
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
        print("saved", name, f"(locations={len(grid_lat)})")


if __name__ == "__main__":
    main()
