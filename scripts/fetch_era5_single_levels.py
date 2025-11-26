#!/usr/bin/env python
"""
ERA5 single-levels fetcher with optional *direct-to-S3* upload & manifest.

- New flags:
    --to-s3           : upload chunk files to S3 and skip local merge
    --s3-subdir STR   : S3 subdir under WEATHER_S3_PREFIX (default: era5-single/<TILE>)
"""
from __future__ import annotations

import os, argparse, datetime as dt, time, sys, calendar, logging
import cdsapi

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

ZIP_MAGIC = b"PK\x03\x04"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
NETCDF_MAGIC_PREFIXES = (b"CDF",)


def log(*args):
    print("[era5-single]", *args, flush=True)


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


def months_between(t0: dt.datetime, t1: dt.datetime):
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


def days_between(t0: dt.datetime, t1: dt.datetime):
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
    end_inc = b - dt.timedelta(seconds=1)
    years = [f"{y:04d}" for y in range(a.year, end_inc.year + 1)]
    hours = [f"{h:02d}:00" for h in range(24)]
    if a.year == end_inc.year and a.month == end_inc.month:
        months = [f"{a.month:02d}"]
        import calendar

        ndays = calendar.monthrange(a.year, a.month)[1]
        first_day = a.day
        last_day = min(end_inc.day, ndays)
        days = [f"{d:02d}" for d in range(first_day, last_day + 1)]
    else:
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
    return {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": years,
        "month": months,
        "day": days,
        "time": hours,
        "area": [args.lat_max, args.lon_min, args.lat_min, args.lon_max],  # N, W, S, E
        "format": "netcdf",
        "expver": ["1", "5"],
    }


def try_retrieve(
    c: cdsapi.Client,
    collection: str,
    request: dict,
    target: str,
    max_retries: int,
    backoff: float,
):
    for k in range(max_retries + 1):
        try:
            if os.path.exists(target) and os.path.getsize(target) < 1024:
                try:
                    os.remove(target)
                except Exception:
                    pass
            c.retrieve(collection, request, target)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                return True, ""
            if k == max_retries:
                return False, "badfile"
            sleep_s = backoff * (2**k)
            log(f"warn: invalid file, retry {k+1}/{max_retries} in {sleep_s:.1f}s")
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


# ---- Shared helpers for S3-direct flow ----
import tempfile, json, shutil

try:
    from utils.s3_utils import s3_enabled, upload_file, upload_bytes, object_exists, download_json, list_object_keys
except Exception:
    from s3_utils import s3_enabled, upload_file, upload_bytes, object_exists, download_json, list_object_keys  # type: ignore

ZIP_MAGIC = b"PK\x03\x04"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
NETCDF_MAGIC_PREFIXES = (b"CDF",)


def _detect_ext(path: str) -> str:
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        if head.startswith(ZIP_MAGIC):
            return ".zip"
        if head.startswith(HDF5_MAGIC) or any(
            head.startswith(p) for p in NETCDF_MAGIC_PREFIXES
        ):
            return ".nc"
    except Exception:
        pass
    return os.path.splitext(path)[1] or ""


def _infer_tile_id_from_outfile(outfile: str) -> str:
    base = os.path.basename(outfile)
    if base.endswith(".nc"):
        base = base[:-3]
    if "_era5" in base:
        return base.split("_era5")[0]
    return base or "tile"


def _s3_subdir_for(
    dataset_short: str, tile_id: str | None, override: str | None
) -> str:
    if override:
        return override.strip("/")
    if tile_id:
        return f"{dataset_short}/{tile_id}"
    return dataset_short


def _manifest_name(outfile: str) -> str:
    stem = os.path.splitext(os.path.basename(outfile))[0]
    return f"{stem}.manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--outfile", type=str, default="tiles/TILE_era5_single.nc")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=5.0)
    ap.add_argument("--granularity", choices=["auto", "month", "day"], default="auto")
    ap.add_argument("--debug-merge", action="store_true")

    # S3 mode
    ap.add_argument(
        "--to-s3",
        action="store_true",
        default=False,
        help="Upload chunk files directly to S3 and skip local merge",
    )
    ap.add_argument(
        "--s3-subdir",
        type=str,
        default=None,
        help="Override S3 subdir (default: era5-single/<TILE>)",
    )

    args = ap.parse_args()

    use_s3 = args.to_s3 or s3_enabled()
    if args.debug_merge and not use_s3:
        logging.getLogger("xarray").setLevel(logging.DEBUG)

    c = cdsapi.Client()

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)

    tile_id = _infer_tile_id_from_outfile(args.outfile)
    subdir = _s3_subdir_for("era5-single", tile_id, args.s3_subdir)
    manifest_parts = []

    chunks = months_between(t0, t1)

    if use_s3:
        # Check if manifest already exists
        mname = _manifest_name(args.outfile)
        manifest_exists = object_exists("manifests/era5-single", mname)
        
        if manifest_exists:
            log(f"[skip] manifest already exists: manifests/era5-single/{mname}")
            # Load existing manifest to check if we need to download any missing parts
            existing_meta = download_json("manifests/era5-single", mname)
            if existing_meta:
                existing_parts = {p.get("tag"): p.get("key") for p in existing_meta.get("parts", [])}
                log(f"[info] existing manifest has {len(existing_parts)} parts")
                
                # Check if all required chunks are present
                required_tags = set()
                for am, bm, tagm in chunks:
                    required_tags.add(tagm)
                    # If month fails, we'll need day tags
                    for ad, bd, tagd in days_between(am, bm):
                        required_tags.add(tagd)
                
                missing_tags = required_tags - set(existing_parts.keys())
                if not missing_tags:
                    log(f"[skip] all required parts already exist in manifest")
                    return
                log(f"[info] need to download {len(missing_tags)} missing parts")
            else:
                log(f"[warn] could not load existing manifest, will re-download")
                manifest_exists = False
        
        import tempfile

        tempdir = tempfile.mkdtemp(prefix="era5_single_dl_")
        try:
            # Load existing manifest if it exists to avoid re-downloading parts
            existing_parts = {}
            if manifest_exists:
                existing_meta = download_json("manifests/era5-single", mname)
                if existing_meta:
                    existing_parts = {p.get("tag"): p.get("key") for p in existing_meta.get("parts", [])}
                    manifest_parts = existing_meta.get("parts", [])[:]  # Copy existing parts
                else:
                    log(f"[warn] could not load existing manifest, starting fresh")
                    existing_parts = {}
                    manifest_parts = []
            else:
                # Manifest doesn't exist - scan S3 for existing part files to resume
                log(f"[info] manifest not found, scanning S3 for existing parts...")
                manifest_parts = []
                existing_parts = {}
                
                # Get the base name pattern for parts
                base_pattern = os.path.splitext(os.path.basename(args.outfile))[0]
                part_prefix = f"{base_pattern}.part_"
                
                # List all existing part files in the subdir
                existing_keys = list_object_keys(subdir, prefix=part_prefix)
                log(f"[info] found {len(existing_keys)} existing part files on S3")
                
                # Parse tags from existing files and add to manifest_parts
                for key in existing_keys:
                    # Extract tag from filename like "TILE_era5_single.part_202201.nc" -> "202201"
                    name = os.path.basename(key)
                    if ".part_" in name:
                        # Try to extract tag (could be month YYYYMM or day YYYYMMDD)
                        tag_part = name.split(".part_")[1]
                        # Remove extension
                        tag = os.path.splitext(tag_part)[0]
                        if tag:
                            existing_parts[tag] = key
                            manifest_parts.append({"tag": tag, "key": key})
                            log(f"[resume] found existing part: {tag} -> {key}")
                
                if existing_parts:
                    log(f"[resume] resuming with {len(existing_parts)} existing parts from S3")
            
            for am, bm, tagm in chunks:
                tag = tagm
                base_name = (
                    os.path.splitext(os.path.basename(args.outfile))[0] + f".part_{tag}"
                )
                tmp_target = os.path.join(tempdir, base_name + ".nc")

                req = build_request(am, bm, args)
                log(
                    f"request month {tagm}  area(N,W,S,E)=({args.lat_max},{args.lon_min},{args.lat_min},{args.lon_max})"
                )
                ok, err = try_retrieve(
                    c,
                    "reanalysis-era5-single-levels",
                    req,
                    tmp_target,
                    args.max_retries,
                    args.backoff,
                )
                if not ok and (err == "cost" or args.granularity in ("auto",)):
                    log("month too large; splitting into days…")
                    for ad, bd, tagd in days_between(am, bm):
                        tag = tagd
                        base_name = (
                            os.path.splitext(os.path.basename(args.outfile))[0]
                            + f".part_{tag}"
                        )
                        tmp_target = os.path.join(tempdir, base_name + ".nc")
                        reqd = build_request(ad, bd, args)
                        log(f"request day {tagd}")
                        okd, errd = try_retrieve(
                            c,
                            "reanalysis-era5-single-levels",
                            reqd,
                            tmp_target,
                            args.max_retries,
                            args.backoff,
                        )
                        if not okd:
                            if errd == "cost":
                                log(
                                    f"day {tagd} too large; reduce bbox/variables or split hours"
                                )
                            else:
                                log(f"error on {tagd}: {errd}")
                            sys.exit(2)
                        ext = _detect_ext(tmp_target)
                        up_name = base_name + ext
                        # Check if this part already exists (either in existing manifest or on S3)
                        if tagd in existing_parts:
                            log(f"[skip] part {tagd} already in manifest")
                            os.remove(tmp_target)
                        elif object_exists(subdir, up_name):
                            log(f"[skip] exists s3://.../{subdir}/{up_name}")
                            # Add to manifest if not already there
                            key = f"{subdir}/{up_name}"
                            if tagd not in existing_parts:
                                existing_parts[tagd] = key
                                manifest_parts.append({"tag": tagd, "key": key})
                            os.remove(tmp_target)
                        else:
                            key = upload_file(tmp_target, subdir=subdir, key=None)
                            log(f"[ok] uploaded s3://.../{key}")
                            manifest_parts.append({"tag": tagd, "key": key})
                            os.remove(tmp_target)
                    continue

                if ok:
                    ext = _detect_ext(tmp_target)
                    up_name = base_name + ext
                    # Check if this part already exists (either in existing manifest or on S3)
                    if tagm in existing_parts:
                        log(f"[skip] part {tagm} already in manifest")
                        os.remove(tmp_target)
                    elif object_exists(subdir, up_name):
                        log(f"[skip] exists s3://.../{subdir}/{up_name}")
                        # Add to manifest if not already there
                        key = f"{subdir}/{up_name}"
                        if tagm not in existing_parts:
                            existing_parts[tagm] = key
                            manifest_parts.append({"tag": tagm, "key": key})
                        os.remove(tmp_target)
                    else:
                        key = upload_file(tmp_target, subdir=subdir, key=None)
                        log(f"[ok] uploaded s3://.../{key}")
                        manifest_parts.append({"tag": tagm, "key": key})
                        os.remove(tmp_target)
                else:
                    log(f"error: {err}")
                    sys.exit(2)

            # Emit/update manifest (only if we have parts)
            if manifest_parts:
                meta = {
                    "dataset": "reanalysis-era5-single-levels",
                    "tile_id": tile_id,
                    "area": [args.lat_max, args.lon_min, args.lat_min, args.lon_max],
                    "start": args.start,
                    "end": args.end,
                    "variables": VARS,
                    "parts": manifest_parts,
                }
                mname = _manifest_name(args.outfile)
                mkey = upload_bytes(
                    json.dumps(meta).encode("utf-8"),
                    subdir="manifests/era5-single",
                    name=mname,
                    content_type="application/json",
                )
                print(f"[manifest] s3://.../{mkey}")
        finally:
            try:
                os.rmdir(tempdir)
            except Exception:
                pass
        return

    # ----- Local (original) path: keep existing behavior (download + merge) -----
    # In local mode you can continue to use your existing script's merge mechanics.
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    from fetch_era5_single_levels import merge_parts  # type: ignore  # placeholder


if __name__ == "__main__":
    main()
