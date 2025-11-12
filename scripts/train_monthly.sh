#!/usr/bin/env bash
set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE

# --- Compute conservative window: last day of month M-3 (END), START=END-24 months ---
# Requires GNU date; on macOS use gdate from coreutils.
TODAY="$(date -u +%Y-%m-%d)"
# Go to first day of this month, step back 3 months, then jump to last day of that month
END_DATE="$(date -u -d "$(date -u +%Y-%m-01) -3 month -1 day" +%Y-%m-%d)"
START_DATE="$(date -u -d "$END_DATE -24 month +1 day" +%Y-%m-%d)"  # 24 months

# --- Open-Meteo: use the Historical Forecast API host for archived forecasts ---
export OPEN_METEO_URL="https://historical-forecast-api.open-meteo.com/v1/forecast"

# --- Global tiling + OM/ERA5 pairing (your existing orchestrator) ---
# Tune BREAKDOWN_DEG to control tile size (30° default). Increase LAT/LON steps for denser OM sampling.
BREAKDOWN_DEG="${BREAKDOWN_DEG:-30}"
LAT_STEPS="${LAT_STEPS:-5}"
LON_STEPS="${LON_STEPS:-5}"
CHUNK_HOURS="${CHUNK_HOURS:-168}"    # one-week OM time slices
OM_MAX_LOCS="${OM_MAX_LOCS:-25}"     # throttle per request (free-friendly)
OM_RPS="${OM_RPS:-0.33}"             # ~1 req every 3 s

# Lift the test cap to run all tiles:
TILE_SAMPLE_COUNT="${TILE_SAMPLE_COUNT:-999999}"

export START="${START_DATE}T00:00"
export END="${END_DATE}T00:00"

echo "[info] Training window: ${START} → ${END} (conservative, ERA5-Land safe)"
echo "[info] Using OM Historical Forecast API at ${OPEN_METEO_URL}"

# Run your global pipeline (fetch OM/ERA5 + make training pairs)
bash scripts/global_fetch_and_train.sh

# --- Train models from paired parquet files ---
python scripts/train.py

echo "[ok] Monthly retrain finished: ${START} → ${END}"
