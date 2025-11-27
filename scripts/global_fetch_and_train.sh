#!/usr/bin/env bash
# Global fetch + train pipeline with:
# - 30° coarse tiling (configurable via BREAKDOWN_DEG)
# - First-N-tiles test mode via TILE_SAMPLE_COUNT (default 1)
# - 5x5 OM steps per tile (LAT_STEPS=5, LON_STEPS=5)
# - ERA5 area aligned to tile bounds
#
# Outputs:
#   - data/om_baseline/omifs_*.json         (or S3 equivalent)
#   - tiles/LAT<i}_LON<j>_era5_single.nc    (if not using S3-only mode)
#   - tiles/LAT<i}_LON<j>_era5_land.nc
#   - data/train_tiles/<TILE>__*.parquet    (also mirrored to S3)
#
# Assumes these scripts exist (from your repo):
#   scripts/fetch_openmeteo_hindcast.py
#   scripts/fetch_era5_single_levels.py
#   scripts/fetch_era5_land.py
#   scripts/make_training_pairs_tile.py

set -uo pipefail
# Note: We don't use 'set -e' globally because we want to continue processing
# tiles even if one fails. Individual commands handle errors via run_logged.
export HDF5_USE_FILE_LOCKING=FALSE

# --- Required: point to your S3 bucket/prefix (AWS creds must be configured) ---
export WEATHER_S3_BUCKET="${WEATHER_S3_BUCKET:-zeus-dataset}"
export WEATHER_S3_PREFIX="${WEATHER_S3_PREFIX:-weather-plus}"

# -----------------------------
# Helper functions (must be defined before use)
# -----------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# -----------------------------
# Time window for data fetch and training
# -----------------------------
# Default START: 2022-01-01 (ERA5 data available from this date)
# Default END: 3 months before current date (ERA5-Land has ~3 month delay)
# 
# For initial bulk fetch (2022-2025), you can override:
#   export START="2022-01-01T00:00"
#   export END="2024-09-30T23:00"  # 3 months before late 2024
#
# For ongoing monthly retraining, use train_monthly.sh which computes
# conservative dates automatically (END = 3 months ago, START = END - 24 months)

if [[ -z "${START:-}" ]]; then
    START="2022-01-01T00:00"
fi

if [[ -z "${END:-}" ]]; then
    # Default to 3 months before current date (ERA5-Land availability)
    # This ensures we only fetch data that's definitely available
    END_DATE="$(date -u -d "$(date -u +%Y-%m-01) -3 month -1 day" +%Y-%m-%d)"
    END="${END_DATE}T23:00"
    log "[info] END not set, using conservative default: ${END} (3 months before today)"
else
    log "[info] Using provided END: ${END}"
fi

log "[info] Data fetch window: ${START} → ${END}"

CHUNK_HOURS="${CHUNK_HOURS:-168}"   # OM batch window

# -----------------------------
# Tiling
# -----------------------------
BREAKDOWN_DEG="${BREAKDOWN_DEG:-30}"

# Build degree edges first to calculate total tiles
LAT_EDGES_TEMP=( $(build_edges -90 90 "$BREAKDOWN_DEG") )
LON_EDGES_TEMP=( $(build_edges -180 180 "$BREAKDOWN_DEG") )

# Process only the first N tiles for testing (default: all tiles).
# Set TILE_SAMPLE_COUNT to a number to limit processing (e.g. 1 for testing).
# For full globe, leave unset or set to a large number (e.g. 999999).
if [[ -z "${TILE_SAMPLE_COUNT:-}" ]]; then
    # Calculate total number of tiles
    TOTAL_TILES=$(( (${#LAT_EDGES_TEMP[@]} - 1) * (${#LON_EDGES_TEMP[@]} - 1) ))
    TILE_SAMPLE_COUNT=$TOTAL_TILES
    log "[info] TILE_SAMPLE_COUNT not set, processing all $TOTAL_TILES tiles"
else
    TILE_SAMPLE_COUNT="${TILE_SAMPLE_COUNT}"
fi

# Optional include/exclude regex for tile names (applied after enumeration)
TILE_INCLUDE_REGEX="${TILE_INCLUDE_REGEX:-}"   # e.g. "^LAT0_LON0$"
TILE_EXCLUDE_REGEX="${TILE_EXCLUDE_REGEX:-}"

# -----------------------------
# Open-Meteo request shape
# -----------------------------
LAT_STEPS="${LAT_STEPS:-5}"
LON_STEPS="${LON_STEPS:-5}"

OM_MAX_LOCS="${OM_MAX_LOCS:-25}"
OM_RPS="${OM_RPS:-0.33}"
OM_RETRIES_429="${OM_RETRIES_429:-6}"
OM_TIMEOUT="${OM_TIMEOUT:-180}"

# -----------------------------
# Script paths
# -----------------------------
OM_FETCH_SCRIPT="${OM_FETCH_SCRIPT:-scripts/fetch_openmeteo_hindcast.py}"
ERA5_SINGLE_SCRIPT="${ERA5_SINGLE_SCRIPT:-scripts/fetch_era5_single_levels.py}"
ERA5_LAND_SCRIPT="${ERA5_LAND_SCRIPT:-scripts/fetch_era5_land.py}"
PAIR_SCRIPT="${PAIR_SCRIPT:-scripts/make_training_pairs_tile.py}"

# Modes
DRY_RUN="${DRY_RUN:-0}"
ERA5_ONLY="${ERA5_ONLY:-0}"
OM_ONLY="${OM_ONLY:-0}"
CLEAN_INTERMEDIATES="${CLEAN_INTERMEDIATES:-0}"

# I/O
OUT_OM_DIR="data/om_baseline"
OUT_TILE_DIR="tiles"
OUT_PAIR_DIR="data/train_tiles"
mkdir -p "$OUT_OM_DIR" "$OUT_TILE_DIR" "$OUT_PAIR_DIR" logs

run_logged() {
    local cmd="$1"
    local logfile="$2"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "[dry-run] $cmd"
        echo "[$(ts)] [dry-run] $cmd" >> "$logfile"
        return 0
    else
        # Run command and capture exit code
        # Note: We don't use 'set -e' globally, so we can capture errors here
        bash -lc "$cmd" 2>&1 | tee -a "$logfile"
        local exit_code=${PIPESTATUS[0]}
        return $exit_code
    fi
}

# Build degree edges: [-90, -60, -30, 0, 30, 60, 90] for 30°
build_edges() {
    local min="$1" max="$2" step="$3"
  python - "$min" "$max" "$step" <<'PY'
import sys
mn=float(sys.argv[1]); mx=float(sys.argv[2]); st=float(sys.argv[3])
edges=[]
x=mn
while x<mx:
    edges.append(x)
    x+=st
edges.append(mx)
print(" ".join(str(int(e)) if float(e).is_integer() else str(e) for e in edges))
PY
}

iso_to_tag() {
    local iso="$1"
    if date -d "$iso" "+%Y%m%d%H" >/dev/null 2>&1; then
        date -d "$iso" "+%Y%m%d%H"
    else
    python - "$iso" <<'PY'
import sys, datetime as dt
print(dt.datetime.fromisoformat(sys.argv[1]).strftime("%Y%m%d%H"))
PY
    fi
}

first_chunk_end_tag() {
  python - "$START" "$CHUNK_HOURS" <<'PY'
import sys, datetime as dt
t0=dt.datetime.fromisoformat(sys.argv[1]); ch=int(sys.argv[2])
print((t0+dt.timedelta(hours=ch)).strftime("%Y%m%d%H"))
PY
}

have_era5_files() {
    local tile="$1"
    
    # Use Python/boto3 to check S3 manifests if S3 is enabled
    if [[ -n "${WEATHER_S3_BUCKET:-}" && -n "${WEATHER_S3_PREFIX:-}" ]]; then
        python - <<PY
import os, sys
os.environ.setdefault("WEATHER_S3_BUCKET", "${WEATHER_S3_BUCKET}")
os.environ.setdefault("WEATHER_S3_PREFIX", "${WEATHER_S3_PREFIX}")
try:
    from utils.s3_utils import object_exists
    single_exists = object_exists("manifests/era5-single", "${tile}_era5_single.manifest.json")
    land_exists = object_exists("manifests/era5-land", "${tile}_era5_land.manifest.json")
    sys.exit(0 if (single_exists and land_exists) else 1)
except Exception as e:
    # Fallback: check local files
    import os.path
    single_local = os.path.exists("${OUT_TILE_DIR}/${tile}_era5_single.nc") and os.path.getsize("${OUT_TILE_DIR}/${tile}_era5_single.nc") > 0
    land_local = os.path.exists("${OUT_TILE_DIR}/${tile}_era5_land.nc") and os.path.getsize("${OUT_TILE_DIR}/${tile}_era5_land.nc") > 0
    sys.exit(0 if (single_local and land_local) else 1)
PY
        return $?
    fi
    
    # Fallback to local NetCDF tiles (legacy behaviour)
    [[ -s "${OUT_TILE_DIR}/${tile}_era5_single.nc" && -s "${OUT_TILE_DIR}/${tile}_era5_land.nc" ]]
}
have_pairs_done() { local tile="$1"; [[ -s "${OUT_TILE_DIR}/${tile}.pairs.done" ]]; }

# -----------------------------
# Tiling enumeration
# -----------------------------
# Use the edges we calculated earlier
LAT_EDGES=( "${LAT_EDGES_TEMP[@]}" )
LON_EDGES=( "${LON_EDGES_TEMP[@]}" )

log "Global tiling: BREAKDOWN_DEG=$BREAKDOWN_DEG  tiles_lat=$(( ${#LAT_EDGES[@]} - 1 ))  tiles_lon=$(( ${#LON_EDGES[@]} - 1 ))"
log "Test sample: TILE_SAMPLE_COUNT=$TILE_SAMPLE_COUNT  OM steps: ${LAT_STEPS}x${LON_STEPS}  Window: ${START} → ${END}"

S_TAG="$(iso_to_tag "$START")"
E_TAG="$(first_chunk_end_tag)"

tile_count=0
for ((i=0; i<${#LAT_EDGES[@]}-1; i++)); do
    for ((j=0; j<${#LON_EDGES[@]}-1; j++)); do
        TILE="LAT${i}_LON${j}"
        LAT_MIN="${LAT_EDGES[$i]}"; LAT_MAX="${LAT_EDGES[$i+1]}"
        LON_MIN="${LON_EDGES[$j]}"; LON_MAX="${LON_EDGES[$j+1]}"
        
        # Optional include/exclude filters
        if [[ -n "$TILE_INCLUDE_REGEX" && ! "$TILE" =~ $TILE_INCLUDE_REGEX ]]; then
            continue
        fi
        if [[ -n "$TILE_EXCLUDE_REGEX" && "$TILE" =~ $TILE_EXCLUDE_REGEX ]]; then
            continue
        fi
        
        # Respect sample cap
        if (( tile_count >= TILE_SAMPLE_COUNT )); then
            log "[stop] Reached TILE_SAMPLE_COUNT=$TILE_SAMPLE_COUNT"
            exit 0
        fi
        tile_count=$((tile_count+1))
        
        log "=== Tile $TILE  area=($LAT_MIN,$LON_MIN,$LAT_MAX,$LON_MAX)  ==="
        
        # ------------------- Open-Meteo (features) -------------------
        if [[ "$ERA5_ONLY" != "1" ]]; then
            # Build command with optional S3 flag
            OM_CMD="python \"$OM_FETCH_SCRIPT\" \
            --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
            --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
            --lat-steps \"$LAT_STEPS\" --lon-steps \"$LON_STEPS\" \
            --start \"$START\" --end \"$END\" \
            --chunk-hours \"$CHUNK_HOURS\" \
            --max-locs \"$OM_MAX_LOCS\" \
            --rps \"$OM_RPS\" \
            --retries-429 \"$OM_RETRIES_429\" \
            --timeout \"$OM_TIMEOUT\""
            
            # Add S3 flag if bucket is configured
            if [[ -n "${WEATHER_S3_BUCKET:-}" ]]; then
                OM_CMD="$OM_CMD --to-s3"
            fi
            
            run_logged "$OM_CMD" "logs/${TILE}_om.log" || log "[warn] OM fetch had errors for $TILE, continuing..."
        else
            log "[om] skipped (ERA5_ONLY=1)"
        fi
        
        # ------------------- ERA5 (targets) -------------------
        if [[ "$OM_ONLY" != "1" ]]; then
            # Always call ERA5 fetch scripts - they will check for existing data
            # and only download missing parts. This allows incremental updates
            # when train_monthly.sh runs with a sliding window.
            # Build ERA5 single command with optional S3 flag
            ERA5_SINGLE_CMD="python \"$ERA5_SINGLE_SCRIPT\" \
            --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
            --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
            --start \"$START\" --end \"$END\" \
            --debug-merge \
            --outfile \"${OUT_TILE_DIR}/${TILE}_era5_single.nc\""
            
            # Build ERA5 land command with optional S3 flag
            ERA5_LAND_CMD="python \"$ERA5_LAND_SCRIPT\" \
            --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
            --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
            --start \"$START\" --end \"$END\" \
            --debug-merge \
            --outfile \"${OUT_TILE_DIR}/${TILE}_era5_land.nc\""
            
            # Add S3 flag if bucket is configured
            if [[ -n "${WEATHER_S3_BUCKET:-}" ]]; then
                ERA5_SINGLE_CMD="$ERA5_SINGLE_CMD --to-s3"
                ERA5_LAND_CMD="$ERA5_LAND_CMD --to-s3"
            fi
            
            # Run ERA5 scripts - continue even if one fails
            if ! run_logged "$ERA5_SINGLE_CMD" "logs/${TILE}_era5_single.log"; then
                log "[warn] ERA5 single fetch failed for $TILE, continuing..."
            fi
            if ! run_logged "$ERA5_LAND_CMD" "logs/${TILE}_era5_land.log"; then
                log "[warn] ERA5 land fetch failed for $TILE, continuing..."
            fi
            
            # ------------------- Pairing (OM→features, ERA5→target) -------------------
            if ! have_pairs_done "$TILE"; then
                # Use S3-hosted ERA5 manifests; pairing code will stream NetCDF parts from S3.
                S3_SINGLE_MANIFEST="s3://${WEATHER_S3_BUCKET}/${WEATHER_S3_PREFIX}/manifests/era5-single/${TILE}_era5_single.manifest.json"
                S3_LAND_MANIFEST="s3://${WEATHER_S3_BUCKET}/${WEATHER_S3_PREFIX}/manifests/era5-land/${TILE}_era5_land.manifest.json"
                
                run_logged \
                "python \"$PAIR_SCRIPT\" \
                --era5-single-manifest \"$S3_SINGLE_MANIFEST\" \
                --era5-land-manifest   \"$S3_LAND_MANIFEST\" \
                --tile-id              \"$TILE\"" \
                "logs/${TILE}_pair.log" || log "[warn] Pairing had errors for $TILE, continuing..."
                : > "${OUT_TILE_DIR}/${TILE}.pairs.done"
                if [[ "$CLEAN_INTERMEDIATES" == "1" ]]; then
                    # Left here for backwards-compat; in S3-only mode these files won't exist.
                    rm -f "${OUT_TILE_DIR}/${TILE}_era5_single.nc" "${OUT_TILE_DIR}/${TILE}_era5_land.nc" || true
                fi
            else
                log "[pair] already paired for $TILE"
            fi
        else
            log "[era5/pair] skipped (OM_ONLY=1)"
        fi
        
        log "=== Done $TILE ==="
    done
done

log "All requested tiles completed (processed: $tile_count)."
