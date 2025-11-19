#!/usr/bin/env python
import os, json, argparse, time, datetime as dt, math, requests, random
from typing import Optional

# Support both "utils.*" and local imports for flexibility
try:
    from utils.om_request import om_request
except Exception:
    from om_request import om_request  # type: ignore

try:
    from utils.s3_utils import s3_enabled, upload_bytes, object_exists
except Exception:
    from s3_utils import s3_enabled, upload_bytes, object_exists  # type: ignore

OM_URL = os.getenv("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
LOCAL_OUT_DIR = os.path.join("data", "om_baseline")

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
        wait = max(0, min(wait, 60))  # cap
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

    r = om_request(OM_URL, params=params, timeout=timeout)
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

    # Batching / throttling
    ap.add_argument("--max-locs", type=int, default=int(os.getenv("OM_MAX_LOCS", "50")))
    ap.add_argument("--rps", type=float, default=float(os.getenv("OM_RPS", "0.5")))
    ap.add_argument("--timeout", type=int, default=int(os.getenv("OM_TIMEOUT", "120")))
    ap.add_argument(
        "--retries-429", type=int, default=int(os.getenv("OM_RETRIES_429", "5"))
    )
    ap.add_argument("--jitter", type=float, default=0.3)

    # S3 controls
    ap.add_argument(
        "--to-s3",
        action="store_true",
        default=False,
        help="Force upload to S3 if WEATHER_S3_BUCKET is configured",
    )
    ap.add_argument(
        "--s3-subdir",
        type=str,
        default="om_baseline",
        help="Subdirectory under WEATHER_S3_PREFIX (default om_baseline)",
    )

    args = ap.parse_args()

    # Decide output target
    use_s3 = args.to_s3 or s3_enabled()
    if not use_s3:
        os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

    lat_list = make_linspace(args.lat_min, args.lat_max, args.lat_steps)
    lon_list = make_linspace(args.lon_min, args.lon_max, args.lon_steps)
    grid_lat, grid_lon = make_grid_pairs(lat_list, lon_list)
    total_locs = len(grid_lat)
    print(
        f"[info] OM total locations: {total_locs} (lat_steps={args.lat_steps} x lon_steps={args.lon_steps})"
    )
    print(f"[info] Output: {'S3' if use_s3 else 'local'}")

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    # Keep a lightweight session (requests.adapters handled by om_request)
    session = requests.Session()
    last_time_holder = [0.0]

    uploaded = 0
    skipped = 0

    for a, b in daterange(t0, t1, args.chunk_hours):
        s = a.strftime("%Y-%m-%dT%H:%M")
        e = b.strftime("%Y-%m-%dT%H:%M")
        print(f"[info] time slice {s} -> {e}")

        for lat_chunk, lon_chunk, batch_idx in chunk_pairs(
            grid_lat, grid_lon, args.max_locs
        ):
            batch_size = len(lat_chunk)
            tag = f"{a.strftime('%Y%m%d%H')}_{b.strftime('%Y%m%d%H')}_b{batch_idx:03d}_n{batch_size}"
            fname = f"omifs_{tag}.json"

            if use_s3:
                if object_exists(args.s3_subdir, fname):
                    print(f"[skip] exists s3://.../{args.s3_subdir}/{fname}")
                    skipped += 1
                    continue
            else:
                out_path = os.path.join(LOCAL_OUT_DIR, fname)
                if os.path.exists(out_path):
                    print(f"[skip] exists {out_path}")
                    skipped += 1
                    continue

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
                    waited = _respect_retry_after(resp)
                    if not waited:
                        time.sleep(2.0 + args.jitter)
                    continue
                if resp is None:
                    raise RuntimeError(f"No response for batch {tag}")
                om = resp.json()
                break

            # ECMWF IFS (optional best-effort)
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
                    if resp2 is None:
                        print(
                            f"[warn] ecmwf_ifs025: no response for batch {tag}, skipping"
                        )
                        break
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

            blob = json.dumps(out).encode("utf-8")
            if use_s3:
                key = upload_bytes(
                    blob, args.s3_subdir, fname, content_type="application/json"
                )
                print(f"[ok] uploaded s3://.../{key}")
            else:
                with open(out_path, "w") as f:
                    f.write(blob.decode("utf-8"))
                print(f"[ok] saved {out_path}")
            uploaded += 1
            if args.jitter > 0:
                time.sleep(args.jitter)

    print(
        f"[done] uploaded={uploaded}, skipped={skipped}, dest={'S3' if use_s3 else LOCAL_OUT_DIR}"
    )


if __name__ == "__main__":
    main()
