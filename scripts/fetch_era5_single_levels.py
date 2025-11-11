#!/usr/bin/env python
"""
Robust ERA5 single-levels fetcher with chunking, retries, merging, and ZIP handling.

Variables:
- 10m_u_component_of_wind, 10m_v_component_of_wind
- 100m_u_component_of_wind, 100m_v_component_of_wind
- surface_pressure, total_precipitation
- 2m_temperature, 2m_dewpoint_temperature

Key fixes in this version:
- ZIP extraction uses unique, collision-proof filenames based on the part tag
- Month/day lists respect actual calendar days within [a,b)
- Clearer merge logging with --debug-merge (per-file summaries + xarray internal logs)
- Optional pruning of merged part files via --prune-parts

Requires a valid ~/.cdsapirc for cdsapi.Client.
"""
from __future__ import annotations

import os
import argparse
import datetime as dt
import time
import sys
import calendar
from typing import List, Tuple, Dict, Set, Optional
import shutil
import zipfile
from pathlib import Path
import logging

import cdsapi

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

# Track original ZIP inputs we encountered so we can optionally prune them later
_SEEN_ZIPS: Set[str] = set()


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
    a: dt.datetime, b: dt.datetime
) -> Tuple[List[str], List[str]]:
    """
    Return (months, days) to cover [a, b_excl) within calendar-valid ranges.
    With our chunking, [a,b_excl) should be within one month; handle cross-month defensively.
    """
    end_inc = b - dt.timedelta(seconds=1)
    if a.year == end_inc.year and a.month == end_inc.month:
        months = [f"{a.month:02d}"]
        ndays = calendar.monthrange(a.year, a.month)[1]
        first_day = a.day
        last_day = min(end_inc.day, ndays)
        days = [f"{d:02d}" for d in range(first_day, last_day + 1)]
    else:
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

    _SEEN_ZIPS.add(path)

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
                return [path]
            for m in members:
                extracted = _safe_extract_with_tag(zf, m, dest_dir, tag_stem)
                if extracted is not None:
                    out_paths.append(str(extracted))
    except zipfile.BadZipFile:
        return [path]

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

    # Minimal logger for xarray internals if requested
    if args.debug_merge:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("xarray").setLevel(logging.DEBUG)
        # netCDF or h5netcdf can be noisy; leave at INFO unless you need more:
        logging.getLogger("netCDF4").setLevel(logging.INFO)
        logging.getLogger("h5netcdf").setLevel(logging.INFO)

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
    merged, used_parts = merge_parts(part_paths, args.outfile, debug=args.debug_merge)
    if merged:
        log(f"Merged -> {args.outfile}")
        if args.prune - parts:
            _prune_after_merge(used_parts)
    else:
        log(
            "xarray not available or merge failed; kept part files. Install xarray/netCDF4 to auto-merge."
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


# -------- Merge / Diagnostics --------


def _brief_ds_summary(ds, path: str) -> str:
    dims = {k: int(v) for k, v in ds.sizes.items()}
    vars_count = len([v for v in ds.data_vars])
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
    """Scan files to report variable coverage, coord mismatches, and time spans."""
    try:
        import xarray as xr
        import numpy as np
    except Exception:
        return

    all_vars: Set[str] = set()
    per_file_vars: Dict[str, Set[str]] = {}
    time_spans: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    lat_names = set()
    lon_names = set()

    for p in paths:
        try:
            ds = xr.open_dataset(
                p, engine=engines[p], decode_times=True, chunks={}, cache=False
            )
            all_vars |= set(ds.data_vars)
            per_file_vars[p] = set(ds.data_vars)
            if "time" in ds:
                try:
                    v = ds["time"].values
                    time_spans[p] = (str(v.min()), str(v.max()))
                except Exception:
                    time_spans[p] = (None, None)
            for cand in ("latitude", "lat", "y"):
                if cand in ds.coords:
                    lat_names.add(cand)
            for cand in ("longitude", "lon", "x"):
                if cand in ds.coords:
                    lon_names.add(cand)
            if debug:
                log("probe:", _brief_ds_summary(ds, p))
            ds.close()
        except Exception as e:
            log(f"warning: preflight could not open {p}: {e}")

    if debug:
        log(
            f"preflight: union of variables across {len(paths)} files = {sorted(all_vars)}"
        )
        # Any file missing expected VARS?
        for p in paths:
            missing = [v for v in VARS if v not in per_file_vars.get(p, set())]
            if missing:
                log(f"preflight: {os.path.basename(p)} missing variables: {missing}")
        # coord name consistency
        log(
            f"preflight: latitude coord names seen: {sorted(lat_names)}; longitude coord names seen: {sorted(lon_names)}"
        )
        # rough time coverage
        for p, (tmin, tmax) in list(time_spans.items())[:10]:
            log(f"preflight: {os.path.basename(p)} time span: {tmin} -> {tmax}")
        if len(time_spans) > 10:
            log(f"preflight: ... {len(time_spans)-10} more time span lines omitted")


def merge_parts(
    parts: List[str], outfile: str, debug: bool = False
) -> Tuple[bool, List[str]]:
    """
    Robust, batch-wise merge that avoids HDF5 locking stalls and slow graph builds.
    - Opens files one-by-one (fast fail), logs progress.
    - Merges in batches, then merges the batches.
    - Uses override semantics to avoid attribute/coord conflicts.

    Returns (merged_ok, used_part_paths).
    """
    if not parts:
        return False, []

    # Expand any leftover ZIPs first
    expanded: List[str] = []
    for p in parts:
        expanded.extend(_maybe_unzip_to_nc(p))

    # Basic filter
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
        return False, []

    # Helper: open a dataset with tight, consistent options
    def _open_clean(path: str):
        import xarray as xr

        last = None
        for eng in ("netcdf4", "h5netcdf", "scipy"):
            try:
                ds = xr.open_dataset(
                    path,
                    engine=eng,
                    decode_times=True,
                    chunks={},  # avoid dask auto-chunk across many files
                    mask_and_scale=True,
                    use_cftime=None,  # let xarray auto-choose
                    cache=False,
                )
                # normalize/sort time; drop duplicate times (keep first)
                if "time" in ds:
                    try:
                        ds = ds.sortby("time")
                        import numpy as np  # local import

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

    # Pre-open all, isolating bad files quickly and logging progress
    good: List[Tuple[str, str]] = []  # (path, engine)
    still_bad: List[Tuple[str, str]] = []
    try:
        import xarray as xr  # ensure present before loop
    except Exception as e:
        log(f"merge skipped (import error: {e})")
        return False, []

    for k, p in enumerate(parts, 1):
        try:
            ds, eng = _open_clean(p)
            if debug:
                log("open:", _brief_ds_summary(ds, p), f" engine={eng}")
            ds.close()  # probe only
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

    # Preflight diagnostics (variables, coord names, time spans)
    _preflight_report([p for p, _ in good], engines_map, debug=debug)

    # Batch open-and-merge to keep memory and graphs manageable
    BATCH = 20
    batch_files = [good[i : i + BATCH] for i in range(0, len(good), BATCH)]
    batch_paths: List[str] = []

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

    # Final merge of batch outputs (small number of files)
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
        # cleanup batch files
        for b in batch_paths:
            try:
                os.remove(b)
            except Exception:
                pass
        # Return list of good part files we actually used (for pruning)
        return True, [p for p, _ in good]
    except Exception as e:
        log(f"merge skipped at final stage ({e.__class__.__name__}: {e})")
        return False, []


# -------- Pruning --------


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
    # Remove part NetCDFs we actually merged
    for p in used_parts:
        _safe_remove(p)
    # Remove any ZIP inputs that were seen
    for z in list(_SEEN_ZIPS):
        if os.path.exists(z):
            _safe_remove(z)


if __name__ == "__main__":
    main()
