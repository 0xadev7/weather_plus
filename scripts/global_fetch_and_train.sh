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

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE

# -----------------------------
# Time window (default: 2022-01-01 → "now")
# -----------------------------
START="${START:-2022-01-01T00:00}"

# If END not provided, default to current UTC hour
if [[ -z "${END:-}" ]]; then
    END="$(date -u +%Y-%m-%dT%H:00)"
fi

CHUNK_HOURS="${CHUNK_HOURS:-168}"   # OM batch window

# -----------------------------
# Tiling
# -----------------------------
BREAKDOWN_DEG="${BREAKDOWN_DEG:-30}"

# Process only the first N tiles for testing (default: 1 tile).
# For full globe set TILE_SAMPLE_COUNT to a large number, e.g. 999999.
TILE_SAMPLE_COUNT="${TILE_SAMPLE_COUNT:-1}"

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

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

run_logged() {
    local cmd="$1"
    local logfile="$2"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "[dry-run] $cmd"
        echo "[$(ts)] [dry-run] $cmd" >> "$logfile"
    else
        bash -lc "$cmd 2>&1 | tee -a \"$logfile\""
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
    [[ -s "${OUT_TILE_DIR}/${tile}_era5_single.nc" && -s "${OUT_TILE_DIR}/${tile}_era5_land.nc" ]]
}
have_pairs_done() { local tile="$1"; [[ -s "${OUT_TILE_DIR}/${tile}.pairs.done" ]]; }

# -----------------------------
# Tiling enumeration
# -----------------------------
LAT_EDGES=( $(build_edges -90 90 "$BREAKDOWN_DEG") )
LON_EDGES=( $(build_edges -180 180 "$BREAKDOWN_DEG") )

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
            run_logged \
            "python \"$OM_FETCH_SCRIPT\" \
            --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
            --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
            --lat-steps \"$LAT_STEPS\" --lon-steps \"$LON_STEPS\" \
            --start \"$START\" --end \"$END\" \
            --chunk-hours \"$CHUNK_HOURS\" \
            --max-locs \"$OM_MAX_LOCS\" \
            --rps \"$OM_RPS\" \
            --retries-429 \"$OM_RETRIES_429\" \
            --timeout \"$OM_TIMEOUT\"" \
            "logs/${TILE}_om.log"
        else
            log "[om] skipped (ERA5_ONLY=1)"
        fi
        
        # ------------------- ERA5 (targets) -------------------
        if [[ "$OM_ONLY" != "1" ]]; then
            if ! have_era5_files "$TILE"; then
                run_logged \
                "python \"$ERA5_SINGLE_SCRIPT\" \
                --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
                --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
                --start \"$START\" --end \"$END\" \
                --debug-merge \
                --outfile \"${OUT_TILE_DIR}/${TILE}_era5_single.nc\"" \
                "logs/${TILE}_era5_single.log"
                
                run_logged \
                "python \"$ERA5_LAND_SCRIPT\" \
                --lat-min \"$LAT_MIN\" --lat-max \"$LON_MIN\" \
                --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
                --start \"$START\" --end \"$END\" \
                --debug-merge \
                --outfile \"${OUT_TILE_DIR}/${TILE}_era5_land.nc\"" \
                "logs/${TILE}_era5_land.log"
            else
                log "[era5] already fetched for $TILE"
            fi
            
            # ------------------- Pairing (OM→features, ERA5→target) -------------------
            if ! have_pairs_done "$TILE"; then
                run_logged \
                "python \"$PAIR_SCRIPT\" \
                --era5-single \"${OUT_TILE_DIR}/${TILE}_era5_single.nc\" \
                --era5-land   \"${OUT_TILE_DIR}/${TILE}_era5_land.nc\" \
                --tile-id     \"$TILE\"" \
                "logs/${TILE}_pair.log"
                : > "${OUT_TILE_DIR}/${TILE}.pairs.done"
                if [[ "$CLEAN_INTERMEDIATES" == "1" ]]; then
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
