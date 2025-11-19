#!/usr/bin/env bash
set -euo pipefail

# --- Required: point to your S3 bucket/prefix (AWS creds must be configured) ---
export WEATHER_S3_BUCKET="${WEATHER_S3_BUCKET:-zeus-dataset}"
export WEATHER_S3_PREFIX="${WEATHER_S3_PREFIX:-weather-plus}"

# --- Time window ---
START="${START:-2022-01-01T00:00}"
END="${END:-2025-11-01T00:00}"

# --- Parallelism and tiling ---
WORKERS="${WORKERS:-4}"
BREAKDOWN_DEG="${BREAKDOWN_DEG:-30}"
LAT_STEPS="${LAT_STEPS:-5}"
LON_STEPS="${LON_STEPS:-5}"
CHUNK_HOURS="${CHUNK_HOURS:-168}"

# --- OM throttling (tune carefully if you increase WORKERS) ---
export OM_MAX_LOCS="${OM_MAX_LOCS:-25}"
export OM_RPS="${OM_RPS:-0.33}"
export OM_RETRIES_429="${OM_RETRIES_429:-6}"
export OM_TIMEOUT="${OM_TIMEOUT:-180}"

python scripts/parallel_fetch_tiles.py \
  --start "$START" --end "$END" \
  --breakdown-deg "$BREAKDOWN_DEG" \
  --lat-steps "$LAT_STEPS" --lon-steps "$LON_STEPS" \
  --chunk-hours "$CHUNK_HOURS" \
  --max-workers "$WORKERS"
echo "[ok] parallel fetch submitted"
