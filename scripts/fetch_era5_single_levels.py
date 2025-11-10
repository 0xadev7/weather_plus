#!/usr/bin/env python
"""
Robust ERA5 single-levels fetcher with chunking, retries, and merging.

Variables:
- 10m_u_component_of_wind, 10m_v_component_of_wind
- 100m_u_component_of_wind, 100m_v_component_of_wind
- surface_pressure, total_precipitation
- 2m_temperature, 2m_dewpoint_temperature
"""
from __future__ import annotations

import os, argparse, datetime as dt, time, sys
from typing import List, Tuple
import cdsapi
import shutil

# ---------------------------
# Helpers
# ---------------------------

VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
    "2m_temperature",
    "2m_dewpoint_temperature",
]

HOURS_ALL = [f"{h:02d}:00" for h in range(24)]


NETCDF_MAGIC = (b"CDF",)  # classic/64-bit offset start with 'CDF'
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _is_netcdf_or_hdf5(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head.startswith(HDF5_MAGIC) or head.startswith(NETCDF_MAGIC)
    except Exception:
        return False


def _weed_bad_parts(parts: List[str]) -> Tuple[List[str], List[str]]:
    good, bad = [], []
    for p in parts:
        if os.path.getsize(p) == 0 or not _is_netcdf_or_hdf5(p):
            bad.append(p)
        else:
            good.append(p)
    return good, bad


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


def hours_for_range(a: dt.datetime, b: dt.datetime) -> List[str]:
    """Return hour list for [a,b) aligned to whole hours in UTC."""
    # cdsapi takes *hours* list only; requests are inclusive by calendar selection.
    # We include all hours present in the interval [a, b). If a/b are not on hour boundaries,
    # just include all 24 hours for the chunk to simplify (small overfetch is fine).
    # If you must be exact: build a set of hours between a and b and format.
    return HOURS_ALL


def part_name(out: str, tag: str) -> str:
    base, ext = os.path.splitext(out)
    if not ext:
        ext = ".nc"
    return f"{base}.part_{tag}{ext}"


def log(*args):
    print("[era5]", *args, flush=True)


def try_retrieve(
    c: cdsapi.Client,
    collection: str,
    request: dict,
    target: str,
    max_retries: int,
    backoff: float,
) -> Tuple[bool, str]:
    """
    Returns (ok, err_msg). 'cost' for too-large requests. 'badfile' if the
    server responded but the file is not NetCDF/HDF5 after all retries.
    """
    for k in range(max_retries + 1):
        try:
            # Remove any previous tiny/corrupt target so retries don't get fooled
            if os.path.exists(target) and os.path.getsize(target) < 1024:
                try:
                    os.remove(target)
                except Exception:
                    pass

            c.retrieve(collection, request, target)

            # Post-download validation
            if (
                os.path.exists(target)
                and os.path.getsize(target) > 0
                and _is_netcdf_or_hdf5(target)
            ):
                return True, ""

            if k == max_retries:
                return False, "badfile"

            sleep_s = backoff * (2**k)
            log(
                f"warn: downloaded file invalid, retry {k+1}/{max_retries} in {sleep_s:.1f}s"
            )
            time.sleep(sleep_s)

        except Exception as e:
            msg = str(e)
            if (
                ("cost limits exceeded" in msg)
                or ("Request too large" in msg)
                or ("413" in msg)
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


def build_request(a: dt.datetime, b: dt.datetime, args) -> dict:
    years = sorted({f"{y:04d}" for y in range(a.year, b.year + 1)})
    # We pass exact months/days/hours covering [a,b). Simpler: full month/day lists.
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = hours_for_range(a, b)

    return {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": years,
        "month": (
            months if (a.month != b.month or a.year != b.year) else [f"{a.month:02d}"]
        ),
        "day": (
            days
            if (a.date() != (b - dt.timedelta(days=1)).date())
            else [f"{a.day:02d}"]
        ),
        "time": hours,
        "area": [args.lat_max, args.lon_min, args.lat_min, args.lon_max],  # N, W, S, E
        "format": "netcdf",
        "expver": ["1", "5"],  # <--- key line to get ERA5T + consolidated
    }


# ---------------------------
# Main
# ---------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--outfile", type=str, default="data/era5_single_levels.nc")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument(
        "--granularity",
        choices=["auto", "month", "day"],
        default="auto",
        help="Chunk size. 'auto' = month, fallback to day if month too big.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    c = cdsapi.Client()  # requires valid ~/.cdsapirc

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    # Build month chunks first
    chunks_m = months_between(t0, t1)
    part_paths: List[str] = []

    for am, bm, tagm in chunks_m:
        # Skip if a merged part already exists (month-level)
        part_m = part_name(args.outfile, tagm)
        if os.path.exists(part_m) and os.path.getsize(part_m) > 0:
            log(f"skip existing month part {part_m}")
            part_paths.append(part_m)
            continue

        if args.granularity in ("day",):
            # Force day-level immediately
            ok = fetch_days_in_month(c, am, bm, args, part_paths)
            if not ok:
                sys.exit(2)
            continue

        # Try month request
        req = build_request(am, bm, args)
        log(
            f"request month {tagm}  area=({args.lat_min},{args.lat_max},{args.lon_min},{args.lon_max})"
        )
        ok, err = try_retrieve(
            c,
            "reanalysis-era5-single-levels",
            req,
            part_m,
            max_retries=args.max_retries,
            backoff=args.backoff,
        )
        if ok:
            log(f"saved {part_m}")
            part_paths.append(part_m)
            continue

        if err == "cost" or args.granularity == "auto":
            log(f"month {tagm} too large; splitting into days…")
            ok = fetch_days_in_month(c, am, bm, args, part_paths)
            if not ok:
                sys.exit(2)
        else:
            log(f"error: {err}")
            sys.exit(2)

    # Merge parts -> outfile
    merged = merge_parts(part_paths, args.outfile)
    if merged:
        log(f"Merged -> {args.outfile}")
    else:
        log(
            "xarray not available; kept part files. Install xarray/netCDF4 to auto-merge."
        )
    print("Saved", args.outfile if merged else f"{len(part_paths)} part files")


def fetch_days_in_month(
    c: cdsapi.Client, am: dt.datetime, bm: dt.datetime, args, part_paths: List[str]
) -> bool:
    days = days_between(am, bm)
    ok_any = False
    for ad, bd, tagd in days:
        part_d = part_name(args.outfile, tagd)
        if os.path.exists(part_d) and os.path.getsize(part_d) > 0:
            log(f"skip existing day part {part_d}")
            part_paths.append(part_d)
            ok_any = True
            continue
        reqd = build_request(ad, bd, args)
        log(f"request day {tagd}")
        ok, err = try_retrieve(
            c,
            "reanalysis-era5-single-levels",
            reqd,
            part_d,
            max_retries=args.max_retries,
            backoff=args.backoff,
        )
        if ok:
            log(f"saved {part_d}")
            part_paths.append(part_d)
            ok_any = True
        elif err == "cost":
            log(
                f"day {tagd} still too large: consider tightening bbox, reducing variables, or split hours"
            )
            return False
        else:
            log(f"error on {tagd}: {err}")
            return False
    return ok_any


def merge_parts(parts: List[str], outfile: str) -> bool:
    if not parts:
        return False
    parts = sorted({p for p in parts if os.path.exists(p) and os.path.getsize(p) > 0})

    # 1) quick header filter
    parts, bad = _weed_bad_parts(parts)
    for b in bad:
        log(f"warning: {b} is not NetCDF/HDF5 (moved to .bad)")
        shutil.move(b, b + ".bad")

    if not parts:
        return False

    try:
        import xarray as xr

        # 2) pinpoint files that still fail to open
        good_parts = []
        still_bad = []
        for p in parts:
            try:
                xr.open_dataset(p, engine="netcdf4").close()
                good_parts.append(p)
                continue
            except Exception:
                pass
            try:
                xr.open_dataset(p, engine="h5netcdf").close()
                good_parts.append(p)
                continue
            except Exception:
                pass
            try:
                xr.open_dataset(p, engine="scipy").close()
                good_parts.append(p)
                continue
            except Exception as e:
                still_bad.append((p, f"{e.__class__.__name__}: {e}"))

        for p, emsg in still_bad:
            log(f"warning: {p} unreadable by xarray ({emsg}); moved to .bad")
            shutil.move(p, p + ".bad")

        if not good_parts:
            return False

        log(f"merging {len(good_parts)} parts")
        ds = None
        last_err = None
        for eng in ("netcdf4", "h5netcdf", "scipy"):
            try:
                ds = xr.open_mfdataset(
                    good_parts,
                    combine="by_coords",
                    engine=eng,
                    parallel=False,
                    decode_times=True,
                )
                write_engine = eng  # use a matching writer
                break
            except Exception as e:
                last_err = e
                ds = None
        if ds is None:
            raise last_err

        if "time" in ds:
            ds = ds.sortby("time")
        tmp = outfile + ".tmp"
        ds.to_netcdf(tmp, engine=write_engine)
        ds.close()
        os.replace(tmp, outfile)
        return True
    except Exception as e:
        log(f"merge skipped ({e.__class__.__name__}: {e})")
        return False


if __name__ == "__main__":
    main()
