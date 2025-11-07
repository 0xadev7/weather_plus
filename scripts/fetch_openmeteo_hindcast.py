#!/usr/bin/env python
import os, json, argparse, datetime as dt, requests

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


def post_om(lat, lon, hourly, start_hour, end_hour, model=None):
    payload = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "timezone": "UTC",
    }
    if model:
        payload["model"] = model
    r = requests.post(OM_URL, json=payload, timeout=120)
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
    ap.add_argument("--start", type=str, required=True)  # ISO
    ap.add_argument("--end", type=str, required=True)
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
    for a, b in daterange(t0, t1, args.chunk_hours):
        s = a.strftime("%Y-%m-%dT%H:%M")
        e = b.strftime("%Y-%m-%dT%H:%M")
        om = post_om(lat_list, lon_list, args.hourly, s, e, model=None)
        ifs = None
        try:
            ifs = post_om(lat_list, lon_list, args.hourly, s, e, model="ecmwf_ifs")
        except Exception:
            pass
        out = {
            "meta": {"lat": lat_list, "lon": lon_list, "start": s, "end": e},
            "om": om,
            "ifs": ifs,
        }
        name = f"omifs_{a.strftime('%Y%m%d%H')}_{b.strftime('%Y%m%d%H')}.json"
        with open(os.path.join(OUT_DIR, name), "w") as f:
            json.dump(out, f)
        print("saved", name)


if __name__ == "__main__":
    main()
