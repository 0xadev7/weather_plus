#!/usr/bin/env python
"""
Robust ERA5-Land fetcher with month->day chunking, retries, validation, merging, and ZIP handling.
Dataset: reanalysis-era5-land (hourly)

Improvements:
- ZIP extraction uses unique, collision-proof filenames based on the part tag
- Calendar-accurate day lists for month chunks
- Deep xarray merge diagnostics with --debug-merge
- Optional pruning of merged parts/ZIPs with --prune-parts
- Batch-wise combine_by_coords to reduce memory/locking issues
"""

from __future__ import annotations

import os
import argparse
import datetime as dt
import time
import sys
import calendar
import shutil
from typing import List, Tuple, Dict, Set, Optional
import zipfile
from pathlib import Path
import logging

import cdsapi

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

NETCDF_MAGIC_PREFIXES = (b"CDF",)
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"

# ZIPs seen (so we can prune after merge if asked)
_SEEN_ZIPS: Set[str] = set()


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


def part_name(out: str, tag: str) -> str:
    base, ext = os.path.splitext(out)
    if not ext:
        ext = ".nc"
    return f"{base}.part_{tag}{ext}"


def log(*args):
    print("[era5-land]", *args, flush=True)


def _is_netcdf_or_hdf5_or_zip(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if head.startswith(HDF5_MAGIC) or any(
            head.startswith(p) for p in NETCDF_MAGIC_PREFIXES
        ):
            return True
        return head.startswith(ZIP_MAGIC)
    except Exception:
        return False


def _weed_bad_parts(parts: List[str]) -> Tuple[List[str], List[str]]:
    good, bad = [], []
    for p in parts:
        try:
            if os.path.getsize(p) == 0:
                bad.append(p)
                continue
            with open(p, "rb") as f:
                head = f.read(8)
            if head.startswith(HDF5_MAGIC) or any(
                head.startswith(pref) for pref in NETCDF_MAGIC_PREFIXES
            ):
                good.append(p)
            else:
                bad.append(p)
        except Exception:
            bad.append(p)
    return good, bad


def _unique_path(dest_dir: Path, base_name: str) -> Path:
    p = dest_dir / base_name
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    k = 1
    while True:
        cand = dest_dir / f"{stem}__x{k:02d}{suffix}"
        if not cand.exists():
            return cand
        k += 1


def _safe_extract_with_tag(
    zf: zipfile.ZipFile, member: zipfile.ZipInfo, dest_dir: Path, tag_stem: str
) -> Path | None:
    base = os.path.basename(member.filename)
    if not base:
        return None
    base_noext = os.path.splitext(base)[0]
    out_name = f"{tag_stem}__{base_noext}.nc"
    final_path = _unique_path(dest_dir, out_name)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp_extract")
    with zf.open(member, "r") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(tmp_path, final_path)
    return final_path


def _maybe_unzip_to_nc(path: str) -> List[str]:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if not head.startswith(ZIP_MAGIC):
            return [path]
    except Exception:
        return [path]

    _SEEN_ZIPS.add(path)

    out_paths: List[str] = []
    dest_dir = Path(path).parent
    tag_stem = Path(path).stem

    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = [
                m
                for m in zf.infolist()
                if not m.is_dir() and m.filename.lower().endswith(".nc")
            ]
            if not members:
                return [path]
            for m in members:
                extracted = _safe_extract_with_tag(zf, m, dest_dir, tag_stem)
                if extracted is not None:
                    out_paths.append(str(extracted))
    except zipfile.BadZipFile:
        return [path]

    return out_paths if out_paths else [path]


def hours_for_range(_: dt.datetime, __: dt.datetime) -> List[str]:
    return HOURS_ALL


def build_request(a: dt.datetime, b: dt.datetime, args, variables: List[str]) -> dict:
    end_inc = b - dt.timedelta(seconds=1)
    years = [f"{y:04d}" for y in range(a.year, end_inc.year + 1)]
    hours = hours_for_range(a, b)

    if a.year == end_inc.year and a.month == end_inc.month:
        months = [f"{a.month:02d}"]
        ndays = calendar.monthrange(a.year, a.month)[1]
        first_day = a.day
        last_day = min(end_inc.day, ndays)
        days = [f"{d:02d}" for d in range(first_day, last_day + 1)]
    else:
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]

    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": [args.lat_max, args.lon_min, args.lat_min, args.lon_max],  # N,W,S,E
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
                "cost limits exceeded" in msg
                or "Request too large" in msg
                or "413" in msg
                or "Payload Too Large" in msg
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


# ---------- Diagnostics helpers ----------


def _brief_ds_summary(ds, path: str) -> str:
    dims = {k: int(v) for k, v in ds.sizes.items()}
    vars_count = len(list(ds.data_vars))
    coords = list(ds.coords)
    time_info = ""
    if "time" in ds:
        try:
            tmin = str(ds["time"].values.min())
            tmax = str(ds["time"].values.max())
            time_info = f" time[{tmin} -> {tmax}]"
        except Exception:
            time_info = " time[?]"
    return f"{os.path.basename(path)} dims={dims} vars={vars_count} coords={coords}{time_info}"


def _preflight_report(paths: List[str], engines: Dict[str, str], debug: bool):
    try:
        import xarray as xr
        import numpy as np  # noqa
    except Exception:
        return
    all_vars: Set[str] = set()
    per_file_vars: Dict[str, Set[str]] = {}
    lat_names, lon_names = set(), set()
    shown = 0
    for p in paths:
        try:
            ds = xr.open_dataset(
                p, engine=engines[p], decode_times=True, chunks={}, cache=False
            )
            all_vars |= set(ds.data_vars)
            per_file_vars[p] = set(ds.data_vars)
            if debug:
                log("open:", _brief_ds_summary(ds, p), f" engine={engines[p]}")
            for cand in ("latitude", "lat", "y"):
                if cand in ds.coords:
                    lat_names.add(cand)
            for cand in ("longitude", "lon", "x"):
                if cand in ds.coords:
                    lon_names.add(cand)
            ds.close()
        except Exception as e:
            log(f"warning: preflight could not open {p}: {e}")
        shown += 1
        if debug and shown % 25 == 0:
            log(f"preflight: scanned {shown}/{len(paths)} files")
    if debug:
        log(f"preflight: union vars = {sorted(all_vars)}")
        for p in paths[:10]:
            missing = [v for v in DEFAULT_VARS if v not in per_file_vars.get(p, set())]
            if missing:
                log(f"preflight: {os.path.basename(p)} missing variables: {missing}")
        log(
            f"preflight: latitude coord names seen: {sorted(lat_names)}; longitude coord names seen: {sorted(lon_names)}"
        )


# ---------- Merge (batch + verbose) ----------


def merge_parts(
    parts: List[str], outfile: str, debug: bool = False
) -> Tuple[bool, List[str]]:
    if not parts:
        return False, []

    # Expand ZIPs
    expanded: List[str] = []
    for p in parts:
        expanded.extend(_maybe_unzip_to_nc(p))

    parts = sorted(
        {p for p in expanded if os.path.exists(p) and os.path.getsize(p) > 0}
    )
    if not parts:
        return False, []

    parts, bad = _weed_bad_parts(parts)
    for b in bad:
        log(f"warning: {b} is not NetCDF/HDF5 (moved to .bad)")
        try:
            shutil.move(b, b + ".bad")
        except Exception:
            pass
    if not parts:
        return False, []

    # Helper to open with fallback engines and clean times
    def _open_clean(path: str):
        import xarray as xr

        last = None
        for eng in ("netcdf4", "h5netcdf", "scipy"):
            try:
                ds = xr.open_dataset(
                    path,
                    engine=eng,
                    decode_times=True,
                    chunks={},  # avoid huge dask graphs
                    mask_and_scale=True,
                    use_cftime=None,
                    cache=False,
                )
                if "time" in ds:
                    try:
                        ds = ds.sortby("time")
                        import numpy as np  # local

                        vals = ds["time"].values
                        _, idx = np.unique(vals, return_index=True)
                        if len(idx) != ds.sizes.get("time", len(idx)):
                            ds = ds.isel(time=sorted(idx))
                    except Exception:
                        pass
                return ds, eng
            except Exception as e:
                last = e
        raise last if last else RuntimeError("failed to open dataset")

    try:
        import xarray as xr  # noqa
    except Exception as e:
        log(f"merge skipped (import error: {e})")
        return False, []

    good: List[Tuple[str, str]] = []
    still_bad: List[Tuple[str, str]] = []

    for k, p in enumerate(parts, 1):
        try:
            ds, eng = _open_clean(p)
            if debug:
                log("probe:", _brief_ds_summary(ds, p), f" engine={eng}")
            ds.close()
            good.append((p, eng))
            if k % 25 == 0:
                log(f"probe opened {k}/{len(parts)} parts OK")
        except Exception as e:
            still_bad.append((p, f"{e.__class__.__name__}: {e}"))

    for p, emsg in still_bad:
        log(f"warning: {p} unreadable; moved to .bad ({emsg})")
        try:
            shutil.move(p, p + ".bad")
        except Exception:
            pass

    if not good:
        return False, []

    engines_map = {p: eng for p, eng in good}
    _preflight_report([p for p, _ in good], engines_map, debug=debug)

    # Batch combine to keep memory reasonable
    BATCH = 20
    batch_files = [good[i : i + BATCH] for i in range(0, len(good), BATCH)]
    batch_paths: List[str] = []

    import xarray as xr  # ensure available

    for bi, items in enumerate(batch_files, 1):
        opened = []
        engines = set()
        for p, eng in items:
            ds, _ = _open_clean(p)
            opened.append(ds)
            engines.add(eng)
        try:
            log(f"merging batch {bi}/{len(batch_files)} with {len(opened)} parts")
            bds = xr.combine_by_coords(
                opened,
                combine_attrs="override",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="override",
            )
            if "time" in bds:
                bds = bds.sortby("time")
            tmp_path = outfile + f".batch_{bi:03d}.nc"
            writer = (
                "netcdf4"
                if "netcdf4" in engines
                else ("h5netcdf" if "h5netcdf" in engines else None)
            )
            if writer:
                bds.to_netcdf(tmp_path, engine=writer)
            else:
                bds.to_netcdf(tmp_path)
            if debug:
                log(f"batch {bi} -> {tmp_path}")
            bds.close()
            for ds in opened:
                ds.close()
            batch_paths.append(tmp_path)
        except Exception as e:
            for ds in opened:
                try:
                    ds.close()
                except Exception:
                    pass
            log(
                f"warning: batch {bi} failed to merge ({e.__class__.__name__}: {e}); moving batch members to .bad"
            )
            for p, _ in items:
                try:
                    shutil.move(p, p + ".bad")
                except Exception:
                    pass

    if not batch_paths:
        return False, []

    # Final merge across batches
    try:
        opened = [
            xr.open_dataset(b, decode_times=True, chunks={}, cache=False)
            for b in batch_paths
        ]
        log(f"final merge of {len(opened)} batch files")
        ds = xr.combine_by_coords(
            opened,
            combine_attrs="override",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            join="override",
        )
        if "time" in ds:
            ds = ds.sortby("time")
        tmp = outfile + ".tmp"
        ds.to_netcdf(tmp, engine="netcdf4")
        ds.close()
        for od in opened:
            try:
                od.close()
            except Exception:
                pass
        os.replace(tmp, outfile)
        for b in batch_paths:
            try:
                os.remove(b)
            except Exception:
                pass
        return True, [p for p, _ in good]
    except Exception as e:
        log(f"merge skipped at final stage ({e.__class__.__name__}: {e})")
        return False, []


# ---------- Pruning ----------


def _safe_remove(p: str):
    try:
        os.remove(p)
        log(f"prune: removed {p}")
    except Exception as e:
        log(f"prune: could not remove {p}: {e}")


def _prune_after_merge(used_parts: List[str]):
    if not used_parts:
        log("prune: nothing to remove (no used parts reported)")
        return
    for p in used_parts:
        _safe_remove(p)
    for z in list(_SEEN_ZIPS):
        if os.path.exists(z):
            _safe_remove(z)


# ---------- Fetch month/day ----------


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
            produced = _maybe_unzip_to_nc(part_d)
            if produced and all(os.path.exists(p) for p in produced):
                log(f"skip existing day part {part_d}")
                part_paths.extend(produced)
                ok_any = True
                continue
        reqd = build_request(ad, bd, args, variables)
        log(f"request day {tagd}")
        ok, err = try_retrieve(
            c, "reanalysis-era5-land", reqd, part_d, max_retries, backoff
        )
        if ok:
            produced = _maybe_unzip_to_nc(part_d)
            part_paths.extend(produced)
            log(f"saved {', '.join(produced)}")
            ok_any = True
        elif err == "cost":
            log("day too large; reduce bbox/variables or split hours")
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
    ap.add_argument(
        "--debug-merge",
        action="store_true",
        help="Enable verbose xarray debug logs and per-file merge diagnostics.",
    )
    ap.add_argument(
        "--prune-parts",
        action="store_true",
        help="After successful merge, delete original part files (and original ZIPs).",
    )
    args = ap.parse_args()

    if args.debug_merge:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("xarray").setLevel(logging.DEBUG)
        logging.getLogger("netCDF4").setLevel(logging.INFO)
        logging.getLogger("h5netcdf").setLevel(logging.INFO)

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    c = cdsapi.Client()

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    variables = args.vars if args.vars else DEFAULT_VARS
    log(f"vars: {variables}")

    parts: List[str] = []
    for am, bm, tagm in months_between(t0, t1):
        part_m = part_name(args.outfile, tagm)
        if os.path.exists(part_m) and os.path.getsize(part_m) > 0:
            produced = _maybe_unzip_to_nc(part_m)
            if produced and all(os.path.exists(p) for p in produced):
                log(f"skip existing month part {part_m}")
                parts.extend(produced)
                continue

        if args.granularity == "day":
            ok = fetch_days_in_month(
                c, am, bm, args, variables, parts, args.max_retries, args.backoff
            )
            if not ok:
                sys.exit(2)
            continue

        reqm = build_request(am, bm, args, variables)
        log(
            f"request month {tagm}  area(N,W,S,E)=({args.lat_max},{args.lon_min},{args.lat_min},{args.lon_max})"
        )
        ok, err = try_retrieve(
            c, "reanalysis-era5-land", reqm, part_m, args.max_retries, args.backoff
        )
        if ok:
            produced = _maybe_unzip_to_nc(part_m)
            parts.extend(produced)
            log(f"saved {', '.join(produced)}")
            continue

        if err == "cost" or args.granularity == "auto":
            log("month too large; splitting into days…")
            ok = fetch_days_in_month(
                c, am, bm, args, variables, parts, args.max_retries, args.backoff
            )
            if not ok:
                sys.exit(2)
        else:
            log(f"error: {err}")
            sys.exit(2)

    merged, used_parts = merge_parts(parts, args.outfile, debug=args.debug_merge)
    if merged:
        log(f"Merged -> {args.outfile}")
        if args.prune_parts:
            _prune_after_merge(used_parts)
        print("Saved", args.outfile)
    else:
        log(
            "xarray not available or parts invalid; kept part files. Install xarray/netCDF4/h5netcdf to auto-merge."
        )
        print(f"Saved {len(parts)} part files")


if __name__ == "__main__":
    main()
