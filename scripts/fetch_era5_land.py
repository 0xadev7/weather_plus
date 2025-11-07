#!/usr/bin/env python
"""
Robust ERA5-Land fetcher with month->day chunking, retries, and merging.
Dataset: reanalysis-era5-land  (hourly)

Default variables (override with --var):
- 2m_temperature
- 2m_dewpoint_temperature
- total_precipitation
- volumetric_soil_water_layer_1..4
- snow_depth
- skin_temperature
"""
from __future__ import annotations

import os, argparse, datetime as dt, time, sys
from typing import List, Tuple
import cdsapi

# ---------------------------
# Defaults (can override with --var ...)
# ---------------------------
DEFAULT_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "snow_depth",
    "skin_temperature",
]
HOURS_ALL = [f"{h:02d}:00" for h in range(24)]


# ---------------------------
# Time helpers
# ---------------------------
def month_start(d: dt.datetime) -> dt.datetime:
    return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(d: dt.datetime) -> dt.datetime:
    y, m = d.year, d.month
    nm = 1 if m == 12 else (m + 1)
    ny = y + 1 if m == 12 else y
    return d.replace(
        year=ny, month=nm, day=1, hour=0, minute=0, second=0, microsecond=0
    )


def day_start(d: dt.datetime) -> dt.datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def next_day(d: dt.datetime) -> dt.datetime:
    return d + dt.timedelta(days=1)


def clamp(a: dt.datetime, lo: dt.datetime, hi: dt.datetime) -> dt.datetime:
    return max(lo, min(a, hi))


def months_between(
    t0: dt.datetime, t1: dt.datetime
) -> List[Tuple[dt.datetime, dt.datetime, str]]:
    res = []
    cur = month_start(t0)
    while cur < t1:
        nxt = next_month(cur)
        a = clamp(cur, t0, t1)
        b = clamp(nxt, t0, t1)
        tag = f"{a.year:04d}{a.month:02d}"
        res.append((a, b, tag))
        cur = nxt
    return res


def days_between(
    t0: dt.datetime, t1: dt.datetime
) -> List[Tuple[dt.datetime, dt.datetime, str]]:
    res = []
    cur = day_start(t0)
    while cur < t1:
        nxt = next_day(cur)
        a = clamp(cur, t0, t1)
        b = clamp(nxt, t0, t1)
        tag = f"{a.year:04d}{a.month:02d}{a.day:02d}"
        res.append((a, b, tag))
        cur = nxt
    return res


# ---------------------------
# IO / logging
# ---------------------------
def part_name(out: str, tag: str) -> str:
    base, ext = os.path.splitext(out)
    if not ext:
        ext = ".nc"
    return f"{base}.part_{tag}{ext}"


def log(*args):
    print("[era5-land]", *args, flush=True)


# ---------------------------
# Request helpers
# ---------------------------
def hours_for_range(_: dt.datetime, __: dt.datetime) -> List[str]:
    return HOURS_ALL  # simple & safe (hourly dataset)


def build_request(a: dt.datetime, b: dt.datetime, args, variables: List[str]) -> dict:
    years = sorted({f"{y:04d}" for y in range(a.year, b.year + 1)})
    months = [f"{m:02d}"]
    days = [f"{d:02d}"]
    hours = hours_for_range(a, b)

    # If the chunk spans >1 month/day, broaden selectors (CDS accepts lists)
    if a.year != b.year or a.month != b.month:
        months = [f"{m:02d}" for m in range(1, 13)]
    if a.date() != (b - dt.timedelta(days=1)).date():
        days = [f"{d:02d}" for d in range(1, 32)]

    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": [args.lat_max, args.lon_min, args.lat_min, args.lon_max],  # N, W, S, E
        "format": "netcdf",
    }


def try_retrieve(
    c: cdsapi.Client,
    collection: str,
    request: dict,
    target: str,
    max_retries: int,
    backoff: float,
) -> Tuple[bool, str]:
    """
    Returns (ok, err). For 403 'cost limits exceeded', caller should split smaller.
    """
    for k in range(max_retries + 1):
        try:
            c.retrieve(collection, request, target)
            return True, ""
        except Exception as e:
            msg = str(e)
            if (
                "cost limits exceeded" in msg
                or "Request too large" in msg
                or "413" in msg
            ):
                return False, "cost"
            if k == max_retries:
                return False, msg
            sleep_s = backoff * (2**k)
            log(
                f"warn: transient error, retry {k+1}/{max_retries} in {sleep_s:.1f}s: {msg}"
            )
            time.sleep(sleep_s)
    return False, "unknown"


# ---------------------------
# Merge
# ---------------------------
def merge_parts(parts: List[str], outfile: str) -> bool:
    if not parts:
        return False
    parts = sorted(
        set(p for p in parts if os.path.exists(p) and os.path.getsize(p) > 0)
    )
    if not parts:
        return False
    try:
        import xarray as xr

        log(f"merging {len(parts)} parts")
        ds = xr.open_mfdataset(parts, combine="by_coords")
        if "time" in ds:
            ds = ds.sortby("time")
        tmp = outfile + ".tmp"
        ds.to_netcdf(tmp)
        ds.close()
        os.replace(tmp, outfile)
        return True
    except Exception as e:
        log(f"merge skipped ({e.__class__.__name__}: {e})")
        return False


# ---------------------------
# Main
# ---------------------------
def fetch_days_in_month(
    c: cdsapi.Client,
    am: dt.datetime,
    bm: dt.datetime,
    args,
    variables: List[str],
    part_paths: List[str],
    max_retries: int,
    backoff: float,
) -> bool:
    ok_any = False
    for ad, bd, tagd in days_between(am, bm):
        part_d = part_name(args.outfile, tagd)
        if os.path.exists(part_d) and os.path.getsize(part_d) > 0:
            log(f"skip existing day part {part_d}")
            part_paths.append(part_d)
            ok_any = True
            continue
        reqd = build_request(ad, bd, args, variables)
        log(f"request day {tagd}")
        ok, err = try_retrieve(
            c, "reanalysis-era5-land", reqd, part_d, max_retries, backoff
        )
        if ok:
            log(f"saved {part_d}")
            part_paths.append(part_d)
            ok_any = True
        elif err == "cost":
            log(f"day {tagd} too large; reduce bbox/variables or split hours")
            return False
        else:
            log(f"error on {tagd}: {err}")
            return False
    return ok_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--outfile", type=str, default="data/era5_land.nc")
    ap.add_argument(
        "--var",
        dest="vars",
        action="append",
        default=None,
        help="Add a variable name (repeatable). If omitted, uses a default set.",
    )
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument(
        "--granularity",
        choices=["auto", "month", "day"],
        default="auto",
        help="Chunking size. 'auto' = month, fallback to day if too large.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    c = cdsapi.Client()  # needs ~/.cdsapirc

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    variables = args.vars if args.vars else DEFAULT_VARS
    log(f"vars: {variables}")

    parts: List[str] = []
    for am, bm, tagm in months_between(t0, t1):
        part_m = part_name(args.outfile, tagm)
        if os.path.exists(part_m) and os.path.getsize(part_m) > 0:
            log(f"skip existing month part {part_m}")
            parts.append(part_m)
            continue

        if args.granularity == "day":
            ok = fetch_days_in_month(
                c, am, bm, args, variables, parts, args.max_retries, args.backoff
            )
            if not ok:
                sys.exit(2)
            continue

        # Try whole month
        reqm = build_request(am, bm, args, variables)
        log(
            f"request month {tagm} area=({args.lat_min},{args.lat_max},{args.lon_min},{args.lon_max})"
        )
        ok, err = try_retrieve(
            c, "reanalysis-era5-land", reqm, part_m, args.max_retries, args.backoff
        )
        if ok:
            log(f"saved {part_m}")
            parts.append(part_m)
            continue

        if err == "cost" or args.granularity == "auto":
            log(f"month {tagm} too large; splitting into days…")
            ok = fetch_days_in_month(
                c, am, bm, args, variables, parts, args.max_retries, args.backoff
            )
            if not ok:
                sys.exit(2)
        else:
            log(f"error: {err}")
            sys.exit(2)

    merged = merge_parts(parts, args.outfile)
    if merged:
        log(f"Merged -> {args.outfile}")
        print("Saved", args.outfile)
    else:
        log(
            "xarray not available; kept part files. Install xarray/netCDF4 to auto-merge."
        )
        print(f"Saved {len(parts)} part files")


if __name__ == "__main__":
    main()
