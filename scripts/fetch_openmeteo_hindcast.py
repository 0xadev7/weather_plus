#!/usr/bin/env python
import os, json, argparse, time, datetime as dt
import math
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


def make_linspace(vmin: float, vmax: float, steps: int):
    if steps <= 1:
        return [vmin]
    return [vmin + i * (vmax - vmin) / (steps - 1) for i in range(steps)]


def make_grid_pairs(lat_list, lon_list):
    lat_seq, lon_seq = [], []
    for lat in lat_list:
        for lon in lon_list:
            lat_seq.append(lat)
            lon_seq.append(lon)
    return lat_seq, lon_seq


def chunk_pairs(lat_seq, lon_seq, max_locs):
    assert len(lat_seq) == len(lon_seq)
    n = len(lat_seq)
    for i in range(0, n, max_locs):
        yield lat_seq[i : i + max_locs], lon_seq[i : i + max_locs], i // max_locs


def _sleep_with_rps(rps, last_time_holder):
    if rps <= 0:
        return
    min_interval = 1.0 / rps
    now = time.time()
    dt_needed = min_interval - (now - last_time_holder[0])
    if dt_needed > 0:
        time.sleep(dt_needed)
    last_time_holder[0] = time.time()


def _respect_retry_after(resp):
    ra = resp.headers.get("Retry-After")
    if not ra:
        return False
    try:
        wait = int(ra)
        wait = max(0, min(wait, 60))  # cap to be nice
        if wait > 0:
            time.sleep(wait)
            return True
    except Exception:
        pass
    return False


def get_om(
    session, lat_seq, lon_seq, hourly, start_hour, end_hour, models=None, timeout=120
):
    if len(lat_seq) != len(lon_seq):
        raise ValueError("latitude and longitude must have the same number of elements")

    params = {
        "latitude": ",".join(f"{x:.6f}" for x in lat_seq),
        "longitude": ",".join(f"{x:.6f}" for x in lon_seq),
        "hourly": ",".join(hourly),
        "start_hour": start_hour,
        "end_hour": end_hour,
        "timezone": "UTC",
        "timeformat": "iso8601",
        "cell_selection": "nearest",
    }
    if models:
        params["models"] = models

    r = session.get(OM_URL, params=params, timeout=timeout)
    # Manually handle 429 to honor Retry-After in caller loop.
    if r.status_code == 429:
        return r  # caller inspects and sleeps
    r.raise_for_status()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--lat-steps", type=int, default=5)
    ap.add_argument("--lon-steps", type=int, default=5)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--hourly", nargs="+", default=DEFAULT_HOURLY)
    ap.add_argument("--chunk-hours", type=int, default=168)

    # New: batching / throttling / robustness knobs for free tier
    ap.add_argument(
        "--max-locs-per-req",
        type=int,
        default=int(os.getenv("OM_MAX_LOCS", "50")),
        help="Max paired (lat,lon) per API call (free plan: keep small, e.g., 25–100).",
    )
    ap.add_argument(
        "--rps",
        type=float,
        default=float(os.getenv("OM_RPS", "0.5")),
        help="Requests per second throttle (e.g., 0.5 -> one request every 2s).",
    )
    ap.add_argument("--timeout", type=int, default=int(os.getenv("OM_TIMEOUT", "120")))
    ap.add_argument(
        "--retries-429",
        type=int,
        default=int(os.getenv("OM_RETRIES_429", "5")),
        help="Max retries if 429 is returned.",
    )
    ap.add_argument(
        "--jitter",
        type=float,
        default=0.3,
        help="Extra random jitter (seconds) added after 429 or between calls.",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    lat_list = make_linspace(args.lat_min, args.lat_max, args.lat_steps)
    lon_list = make_linspace(args.lon_min, args.lon_max, args.lon_steps)
    grid_lat, grid_lon = make_grid_pairs(lat_list, lon_list)
    total_locs = len(grid_lat)
    print(
        f"[info] total locations: {total_locs} "
        f"(lat_steps={args.lat_steps} x lon_steps={args.lon_steps}), "
        f"batching with max_locs_per_req={args.max_locs_per_req}"
    )

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    session = make_retry_session()
    last_time_holder = [0.0]

    for a, b in daterange(t0, t1, args.chunk_hours):
        s = a.strftime("%Y-%m-%dT%H:%M")
        e = b.strftime("%Y-%m-%dT%H:%M")
        print(f"[info] time slice {s} -> {e}")

        for lat_chunk, lon_chunk, batch_idx in chunk_pairs(
            grid_lat, grid_lon, args.max_locs_per_req
        ):
            batch_size = len(lat_chunk)
            tag = f"{a.strftime('%Y%m%d%H')}_{b.strftime('%Y%m%d%H')}_b{batch_idx:03d}_n{batch_size}"
            out_path = os.path.join(OUT_DIR, f"omifs_{tag}.json")

            if os.path.exists(out_path):
                print(f"[skip] exists {out_path}")
                continue

            # Throttle before request
            _sleep_with_rps(args.rps, last_time_holder)

            # Combined models
            tries = 0
            while True:
                tries += 1
                resp = get_om(
                    session,
                    lat_chunk,
                    lon_chunk,
                    args.hourly,
                    s,
                    e,
                    models=None,
                    timeout=args.timeout,
                )
                if isinstance(resp, requests.Response) and resp.status_code == 429:
                    if tries > args.retries_429:
                        raise RuntimeError(f"Too many 429s for batch {tag}")
                    # Respect Retry-After; add small jitter to avoid thundering herd
                    waited = _respect_retry_after(resp)
                    if not waited:
                        time.sleep(2.0 + args.jitter)
                    continue
                om = resp.json()
                break

            # ECMWF IFS (optional, tolerate failures)
            ifs = None
            try:
                _sleep_with_rps(args.rps, last_time_holder)
                tries = 0
                while True:
                    tries += 1
                    resp2 = get_om(
                        session,
                        lat_chunk,
                        lon_chunk,
                        args.hourly,
                        s,
                        e,
                        models="ecmwf_ifs025",
                        timeout=args.timeout,
                    )
                    if (
                        isinstance(resp2, requests.Response)
                        and resp2.status_code == 429
                    ):
                        if tries > args.retries_429:
                            print(
                                f"[warn] ecmwf_ifs025: too many 429s for batch {tag}, skipping"
                            )
                            break
                        waited = _respect_retry_after(resp2)
                        if not waited:
                            time.sleep(2.0 + args.jitter)
                        continue
                    ifs = resp2.json()
                    break
            except Exception as ex:
                print(f"[warn] ecmwf_ifs025 request failed for {tag}: {ex}")

            out = {
                "meta": {
                    "lat_axis": lat_list,
                    "lon_axis": lon_list,
                    "paired_lat": lat_chunk,
                    "paired_lon": lon_chunk,
                    "start": s,
                    "end": e,
                    "hourly": args.hourly,
                    "url_base": OM_URL,
                    "batch_index": batch_idx,
                    "batch_size": batch_size,
                },
                "om": om,
                "ifs": ifs,
            }

            with open(out_path, "w") as f:
                json.dump(out, f)
            print(f"[ok] saved {out_path}")

            # light jitter to spread load
            if args.jitter > 0:
                time.sleep(args.jitter)


if __name__ == "__main__":
    main()
