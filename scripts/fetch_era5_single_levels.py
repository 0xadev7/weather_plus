#!/usr/bin/env python
"""
Robust ERA5 single-levels fetcher with chunking, retries, merging, and ZIP handling.

Variables:
- 10m_u_component_of_wind, 10m_v_component_of_wind
- 100m_u_component_of_wind, 100m_v_component_of_wind
- surface_pressure, total_precipitation
- 2m_temperature, 2m_dewpoint_temperature

Key fixes in this version:
- ZIP extraction now uses unique, collision-proof filenames based on the part tag
- Month/day lists respect actual calendar days within [a,b)
- Clearer merge logging

Requires a valid ~/.cdsapirc for cdsapi.Client.
"""
from __future__ import annotations

import os
import argparse
import datetime as dt
import time
import sys
import calendar
from typing import List, Tuple
import cdsapi
import shutil
import zipfile
from pathlib import Path

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
ZIP_MAGIC = b"PK\x03\x04"


def _is_netcdf_or_hdf5_or_zip(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return (
            head.startswith(HDF5_MAGIC)
            or head.startswith(NETCDF_MAGIC)
            or head.startswith(ZIP_MAGIC)
        )
    except Exception:
        return False


def _weed_bad_parts(parts: List[str]) -> Tuple[List[str], List[str]]:
    """Filter out zero-byte or non NetCDF/HDF5 files. ZIPs should have been expanded earlier."""
    good, bad = [], []
    for p in parts:
        try:
            if os.path.getsize(p) == 0:
                bad.append(p)
                continue
            with open(p, "rb") as f:
                head = f.read(8)
            if head.startswith(HDF5_MAGIC) or head.startswith(NETCDF_MAGIC):
                good.append(p)
            else:
                bad.append(p)
        except Exception:
            bad.append(p)
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
    """Return hour list for [a,b); simplified to all hours for robustness."""
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
    for k in range(max_retries + 1):
        try:
            if os.path.exists(target) and os.path.getsize(target) < 1024:
                try:
                    os.remove(target)
                except Exception:
                    pass

            c.retrieve(collection, request, target)

            if (
                os.path.exists(target)
                and os.path.getsize(target) > 0
                and _is_netcdf_or_hdf5_or_zip(target)
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


def _build_month_day_lists(
    a: dt.datetime, b_excl: dt.datetime
) -> Tuple[List[str], List[str]]:
    """
    Return (months, days) to cover [a, b_excl) within calendar-valid ranges.
    With our chunking, [a,b_excl) should be within one month; handle cross-month defensively.
    """
    end_inc = b_excl - dt.timedelta(seconds=1)
    if a.year == end_inc.year and a.month == end_inc.month:
        months = [f"{a.month:02d}"]
        ndays = calendar.monthrange(a.year, a.month)[1]
        first_day = a.day
        last_day = min(end_inc.day, ndays)
        days = [f"{d:02d}" for d in range(first_day, last_day + 1)]
    else:
        # Very rare for our chunking; fall back to broad but valid selections
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
    return months, days


def build_request(a: dt.datetime, b: dt.datetime, args) -> dict:
    years = [f"{y:04d}" for y in range(a.year, (b - dt.timedelta(seconds=1)).year + 1)]
    hours = hours_for_range(a, b)
    months, days = _build_month_day_lists(a, b)
    return {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": [args.lat_max, args.lon_min, args.lat_min, args.lon_max],  # N, W, S, E
        "format": "netcdf",
        "expver": ["1", "5"],  # may trigger ZIP response with multiple streams
    }


# ---------------------------
# ZIP handling (collision-proof)
# ---------------------------


def _unique_path(dest_dir: Path, base_name: str) -> Path:
    p = dest_dir / base_name
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    k = 1
    while True:
        cand = dest_dir / f"{stem}__x{k:02d}{suffix}"
        if not cand.exists():
            return cand
        k += 1


def _safe_extract_with_tag(
    zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest_dir: Path, tag_stem: str
) -> Path | None:
    """Extract member safely into dest_dir using a unique name including tag_stem."""
    base = os.path.basename(member.filename)
    if not base:
        return None
    # Force .nc extension and inject the part tag to avoid collisions
    base_noext = os.path.splitext(base)[0]
    out_name = f"{tag_stem}__{base_noext}.nc"
    final_path = _unique_path(dest_dir, out_name)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp_extract")
    with zf.open(member, "r") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp_path, final_path)
    return final_path


def _maybe_unzip_to_nc(path: str) -> List[str]:
    """
    If 'path' is a ZIP, extract contained .nc files next to it using a unique, tag-based filename.
    If not a ZIP, return [path].
    """
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if not head.startswith(ZIP_MAGIC):
            return [path]
    except Exception:
        return [path]

    out_paths: List[str] = []
    dest_dir = Path(path).parent
    tag_stem = Path(path).stem  # includes ".part_YYYYMMDD" etc.

    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = [
                m
                for m in zf.infolist()
                if not m.is_dir() and m.filename.lower().endswith(".nc")
            ]
            if not members:
                # keep original so caller can decide
                return [path]
            for m in members:
                extracted = _safe_extract_with_tag(zf, m, dest_dir, tag_stem)
                if extracted is not None:
                    out_paths.append(str(extracted))
    except zipfile.BadZipFile:
        return [path]

    # Keep the original ZIP (which might have .nc extension) for provenance
    return out_paths if out_paths else [path]


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
        part_m = part_name(args.outfile, tagm)
        if os.path.exists(part_m) and os.path.getsize(part_m) > 0:
            log(f"skip existing month part {part_m}")
            part_paths.extend(_maybe_unzip_to_nc(part_m))
            continue

        if args.granularity in ("day",):
            ok = fetch_days_in_month(c, am, bm, args, part_paths)
            if not ok:
                sys.exit(2)
            continue

        # Try month request
        req = build_request(am, bm, args)
        log(
            f"request month {tagm}  area(N,W,S,E)=({args.lat_max},{args.lon_min},{args.lat_min},{args.lon_max})"
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
            produced = _maybe_unzip_to_nc(part_m)
            part_paths.extend(produced)
            log(f"saved {', '.join(produced)}")
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
            part_paths.extend(_maybe_unzip_to_nc(part_d))
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
            produced = _maybe_unzip_to_nc(part_d)
            part_paths.extend(produced)
            log(f"saved {', '.join(produced)}")
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

    expanded: List[str] = []
    for p in parts:
        expanded.extend(_maybe_unzip_to_nc(p))

    parts = sorted(
        {p for p in expanded if os.path.exists(p) and os.path.getsize(p) > 0}
    )
    parts, bad = _weed_bad_parts(parts)
    for b in bad:
        log(f"warning: {b} is not NetCDF/HDF5 (moved to .bad)")
        try:
            shutil.move(b, b + ".bad")
        except Exception:
            pass

    if not parts:
        return False

    try:
        import xarray as xr

        good_parts = []
        still_bad = []
        for p in parts:
            opened = False
            for eng in ("netcdf4", "h5netcdf", "scipy"):
                try:
                    xr.open_dataset(p, engine=eng).close()
                    opened = True
                    break
                except Exception:
                    continue
            if opened:
                good_parts.append(p)
            else:
                try:
                    xr.open_dataset(p).close()
                except Exception as e:
                    still_bad.append((p, f"{e.__class__.__name__}: {e}"))
                else:
                    good_parts.append(p)

        for p, emsg in still_bad:
            log(f"warning: {p} unreadable by xarray ({emsg}); moved to .bad")
            try:
                shutil.move(p, p + ".bad")
            except Exception:
                pass

        if not good_parts:
            return False

        log(f"merging {len(good_parts)} parts")
        ds = None
        last_err = None
        write_engine = None
        for eng in ("netcdf4", "h5netcdf", "scipy"):
            try:
                ds = xr.open_mfdataset(
                    good_parts,
                    combine="by_coords",
                    engine=eng,
                    parallel=False,
                    decode_times=True,
                )
                write_engine = eng
                break
            except Exception as e:
                last_err = e
                ds = None
        if ds is None:
            raise last_err  # type: ignore

        if "time" in ds:
            ds = ds.sortby("time")
        tmp = outfile + ".tmp"
        if write_engine in ("netcdf4", "h5netcdf"):
            ds.to_netcdf(tmp, engine=write_engine)
        else:
            ds.to_netcdf(tmp)
        ds.close()
        os.replace(tmp, outfile)
        return True
    except Exception as e:
        log(f"merge skipped ({e.__class__.__name__}: {e})")
        return False


if __name__ == "__main__":
    main()
