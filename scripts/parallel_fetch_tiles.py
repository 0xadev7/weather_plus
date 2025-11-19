#!/usr/bin/env python
"""
parallel_fetch_tiles.py — Fetch OM + ERA5 for global tiles *in parallel* and upload directly to S3.

Requirements:
- AWS creds available to boto3 (env / profile).
- WEATHER_S3_BUCKET (and optional WEATHER_S3_PREFIX) set.
- cdsapi configured (~/.cdsapirc).

This spawns one process per tile (up to --max-workers). Within each tile, the 3 fetchers
run sequentially to respect provider limits; across tiles, they are parallelized.

Example:
  python scripts/parallel_fetch_tiles.py \
    --start 2025-10-25T00:00 --end 2025-11-01T00:00 \
    --breakdown-deg 30 --lat-steps 5 --lon-steps 5 \
    --max-workers 4
"""
from __future__ import annotations

import argparse, subprocess, math, os, sys, re
from concurrent.futures import ProcessPoolExecutor, as_completed

def build_edges(mn: float, mx: float, step: float):
    edges = []
    x = mn
    while x < mx:
        edges.append(x)
        x += step
    edges.append(mx)
    return edges

def tile_id(i: int, j: int) -> str:
    return f"LAT{i}_LON{j}"

def run_tile(args, i: int, j: int, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    tid = tile_id(i, j)
    env = os.environ.copy()
    cmds = []

    # OM
    if not args.era5_only:
        cmds.append([
            sys.executable, args.om_script,
            "--lat-min", str(lat_min), "--lat-max", str(lat_max),
            "--lon-min", str(lon_min), "--lon-max", str(lon_max),
            "--lat-steps", str(args.lat_steps), "--lon-steps", str(args.lon_steps),
            "--start", args.start, "--end", args.end,
            "--chunk-hours", str(args.chunk_hours),
            "--max-locs", str(args.om_max_locs),
            "--rps", str(args.om_rps),
            "--retries-429", str(args.om_retries_429),
            "--timeout", str(args.om_timeout),
            "--to-s3", "--s3-subdir", args.om_s3_subdir
        ])

    # ERA5 single
    if not args.om_only:
        era5_single_out = f"tiles/{tid}_era5_single.nc"
        cmds.append([
            sys.executable, args.era5_single_script,
            "--lat-min", str(lat_min), "--lat-max", str(lat_max),
            "--lon-min", str(lon_min), "--lon-max", str(lon_max),
            "--start", args.start, "--end", args.end,
            "--outfile", era5_single_out,
            "--to-s3", "--s3-subdir", f"{args.era5_single_s3_subdir}/{tid}"
        ])

        # ERA5 land
        era5_land_out = f"tiles/{tid}_era5_land.nc"
        cmds.append([
            sys.executable, args.era5_land_script,
            "--lat-min", str(lat_min), "--lat-max", str(lat_max),
            "--lon-min", str(lon_min), "--lon-max", str(lon_max),
            "--start", args.start, "--end", args.end,
            "--outfile", era5_land_out,
            "--to-s3", "--s3-subdir", f"{args.era5_land_s3_subdir}/{tid}"
        ])

    for cmd in cmds:
        print("[tile", tid, "]", " ".join(cmd))
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            raise SystemExit(f"Tile {tid} step failed with code {res.returncode}")
    return tid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--breakdown-deg", type=float, default=30.0)
    ap.add_argument("--lat-steps", type=int, default=5)
    ap.add_argument("--lon-steps", type=int, default=5)
    ap.add_argument("--chunk-hours", type=int, default=168)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--tile-include-regex", type=str, default="")
    ap.add_argument("--tile-exclude-regex", type=str, default="")

    ap.add_argument("--om-only", action="store_true")
    ap.add_argument("--era5-only", action="store_true")

    # script paths (allow override)
    ap.add_argument("--om-script", default="scripts/fetch_openmeteo_hindcast.py")
    ap.add_argument("--era5-single-script", default="scripts/fetch_era5_single_levels.py")
    ap.add_argument("--era5-land-script", default="scripts/fetch_era5_land.py")

    # OM knobs
    ap.add_argument("--om-max-locs", type=int, default=int(os.getenv("OM_MAX_LOCS", "25")))
    ap.add_argument("--om-rps", type=float, default=float(os.getenv("OM_RPS", "0.33")))
    ap.add_argument("--om-retries-429", type=int, default=int(os.getenv("OM_RETRIES_429", "6")))
    ap.add_argument("--om-timeout", type=int, default=int(os.getenv("OM_TIMEOUT", "180")))

    # S3 subdirs (under WEATHER_S3_PREFIX)
    ap.add_argument("--om-s3-subdir", default="om_baseline")
    ap.add_argument("--era5-single-s3-subdir", default="era5-single")
    ap.add_argument("--era5-land-s3-subdir", default="era5-land")

    args = ap.parse_args()

    # Edges
    lat_edges = build_edges(-90, 90, args.breakdown_deg)
    lon_edges = build_edges(-180, 180, args.breakdown_deg)

    include_re = re.compile(args.tile_include_regex) if args.tile_include_regex else None
    exclude_re = re.compile(args.tile_exclude_regex) if args.tile_exclude_regex else None

    # Enumerate
    tiles = []
    for i in range(len(lat_edges)-1):
        for j in range(len(lon_edges)-1):
            tid = tile_id(i, j)
            if include_re and not include_re.search(tid):
                continue
            if exclude_re and exclude_re.search(tid):
                continue
            tiles.append((i, j, lat_edges[i], lat_edges[i+1], lon_edges[j], lon_edges[j+1]))

    print(f"[info] tiles: {len(tiles)}  workers={args.max_workers}")
    if not tiles:
        print("[warn] no tiles matched filters")
        return

    ok = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(run_tile, args, i, j, la0, la1, lo0, lo1) for (i, j, la0, la1, lo0, lo1) in tiles]
        for fu in as_completed(futures):
            try:
                tid = fu.result()
                print("[done]", tid)
                ok += 1
            except Exception as e:
                print("[error]", e)

    print(f"[summary] success tiles: {ok}/{len(tiles)}")

if __name__ == "__main__":
    main()
