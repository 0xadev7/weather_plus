#!/usr/bin/env bash
# Global fetch + train pipeline (resume-friendly, OM-free-plan safe)
# Now with SINGLE_TILE test mode (7 days), and global tile loops commented out by default.

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE

# -----------------------------
# Quick-start (7-day test window)
# -----------------------------
START="${START:-2025-10-25T00:00}"
END="${END:-2025-11-01T00:00}"    # 7 days

# Throttle knobs for Open-Meteo (free plan friendly)
export OM_MAX_LOCS="${OM_MAX_LOCS:-25}"
export OM_RPS="${OM_RPS:-0.33}"
export OM_RETRIES_429="${OM_RETRIES_429:-6}"
export OM_TIMEOUT="${OM_TIMEOUT:-180}"

# Scripts
OM_FETCH_SCRIPT="${OM_FETCH_SCRIPT:-scripts/fetch_openmeteo_hindcast.py}"
ERA5_SINGLE_SCRIPT="${ERA5_SINGLE_SCRIPT:-scripts/fetch_era5_single_levels.py}"
ERA5_LAND_SCRIPT="${ERA5_LAND_SCRIPT:-scripts/fetch_era5_land.py}"

# Modes / filters
DRY_RUN="${DRY_RUN:-0}"
ERA5_ONLY="${ERA5_ONLY:-0}"
OM_ONLY="${OM_ONLY:-0}"
CLEAN_INTERMEDIATES="${CLEAN_INTERMEDIATES:-0}"
TILE_INCLUDE_REGEX="${TILE_INCLUDE_REGEX:-}"
TILE_EXCLUDE_REGEX="${TILE_EXCLUDE_REGEX:-}"

# -----------------------------
# SINGLE LOCATION TEST (default)
# -----------------------------
SINGLE_TILE="${SINGLE_TILE:-1}"

# Choose a single point; set min=max so OM grid is exactly one location.
# Example: Lisbon, PT
TEST_LAT="${TEST_LAT:-38.7200}"
TEST_LON="${TEST_LON:- -9.1400}"

# For OM grid, force 1x1 so JSON pairs are a single point
LAT_STEPS="${LAT_STEPS:-1}"
LON_STEPS="${LON_STEPS:-1}"
CHUNK_HOURS="${CHUNK_HOURS:-168}"  # fetch in 7-day chunks

OUT_OM_DIR="data/om_baseline"
OUT_TILE_DIR="tiles"
mkdir -p "$OUT_OM_DIR" "$OUT_TILE_DIR" logs

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

iso_to_tag() {
  local iso="$1"
  if date -d "$iso" "+%Y%m%d%H" >/dev/null 2>&1; then
    date -d "$iso" "+%Y%m%d%H"
  else
    python - "$iso" <<'PY'
import sys, datetime as dt
s=sys.argv[1]
print(dt.datetime.fromisoformat(s).strftime("%Y%m%d%H"))
PY
  fi
}

first_chunk_end_tag() {
  python - "$START" "$CHUNK_HOURS" <<'PY'
import sys, datetime as dt
s=sys.argv[1]; ch=int(sys.argv[2])
t0=dt.datetime.fromisoformat(s); t1=t0+dt.timedelta(hours=ch)
print(t1.strftime("%Y%m%d%H"))
PY
}

have_era5_files() {
  local tile="$1"
  [[ -s "${OUT_TILE_DIR}/${tile}_era5_single.nc" && -s "${OUT_TILE_DIR}/${tile}_era5_land.nc" ]]
}

have_pairs_done() {
  local tile="$1"; [[ -s "${OUT_TILE_DIR}/${tile}.pairs.done" ]]
}

have_om_any_batch_for_timeslice() {
  local start_tag="$1"; local end_tag="$2"
  compgen -G "${OUT_OM_DIR}/omifs_${start_tag}_${end_tag}_b*_n*.json" > /dev/null
}

log "Pipeline start: START=$START END=$END SINGLE_TILE=$SINGLE_TILE"
log "OM knobs: OM_MAX_LOCS=$OM_MAX_LOCS OM_RPS=$OM_RPS OM_RETRIES_429=$OM_RETRIES_429 OM_TIMEOUT=$OM_TIMEOUT"
[[ "$DRY_RUN" == "1" ]] && log "DRY_RUN=1"

S_TAG="$(iso_to_tag "$START")"
E_TAG="$(first_chunk_end_tag)"

if [[ "$SINGLE_TILE" == "1" ]]; then
  # Single-location tile. Set area with min=max so OM grid is exactly the point.
  LAT_MIN="$TEST_LAT"; LAT_MAX="$TEST_LAT"
  LON_MIN="$TEST_LON"; LON_MAX="$TEST_LON"
  TILE="TEST"

  log "=== Single tile $TILE (lat=$TEST_LAT, lon=$TEST_LON) ==="

  if [[ "$ERA5_ONLY" != "1" ]]; then
    have_om_any_batch_for_timeslice "$S_TAG" "$E_TAG" && \
      log "[om] some OM batches exist for first slice; will fill gaps."

    run_logged \
      "python \"$OM_FETCH_SCRIPT\" \
        --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
        --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
        --lat-steps \"$LAT_STEPS\" --lon-steps \"$LON_STEPS\" \
        --start \"$START\" --end \"$END\" \
        --chunk-hours \"$CHUNK_HOURS\"" \
      "logs/${TILE}_om.log"
  else
    log "[om] skipped (ERA5_ONLY=1)"
  fi

  if [[ "$OM_ONLY" != "1" ]]; then
    if ! have_era5_files "$TILE"; then
      run_logged \
        "python \"$ERA5_SINGLE_SCRIPT\" \
          --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
          --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
          --start \"$START\" --end \"$END\" \
          --debug-merge --prune-parts \
          --outfile \"${OUT_TILE_DIR}/${TILE}_era5_single.nc\"" \
        "logs/${TILE}_era5_single.log"

      run_logged \
        "python \"$ERA5_LAND_SCRIPT\" \
          --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
          --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
          --start \"$START\" --end \"$END\" \
          --debug-merge --prune-parts \
          --outfile \"${OUT_TILE_DIR}/${TILE}_era5_land.nc\"" \
        "logs/${TILE}_era5_land.log"
    else
      log "[era5] already fetched for $TILE"
    fi

    if ! have_pairs_done "$TILE"; then
      run_logged \
        "python scripts/make_training_pairs_tile.py \
          --era5-single \"${OUT_TILE_DIR}/${TILE}_era5_single.nc\" \
          --era5-land   \"${OUT_TILE_DIR}/${TILE}_era5_land.nc\" \
          --tile-id     \"$TILE\"" \
        "logs/${TILE}_pair.log"
      : > "${OUT_TILE_DIR}/${TILE}.pairs.done"
      [[ "$CLEAN_INTERMEDIATES" == "1" ]] && rm -f "${OUT_TILE_DIR}/${TILE}_era5_single.nc" "${OUT_TILE_DIR}/${TILE}_era5_land.nc" || true
    else
      log "[pair] already paired for $TILE"
    fi
  else
    log "[era5/pair] skipped (OM_ONLY=1)"
  fi

  log "=== Done $TILE ==="

else
  # -----------------------------
  # GLOBAL (commented by default for testing)
  # -----------------------------

  # LAT_EDGES=("-90" "-60" "-30" "0" "30" "60" "90")
  # LON_EDGES=("-180" "-150" "-120" "-90" "-60" "-30" "0" "30" "60" "90" "120" "150" "180")

  # for ((i=0; i<${#LAT_EDGES[@]}-1; i++)); do
  #   for ((j=0; j<${#LON_EDGES[@]}-1; j++)); do
  #     LAT_MIN=${LAT_EDGES[$i]}; LAT_MAX=${LAT_EDGES[$i+1]}
  #     LON_MIN=${LON_EDGES[$j]}; LON_MAX=${LON_EDGES[$j+1]}
  #     TILE="LAT${i}_LON${j}"
  #     # ... same body as above ...
  #   done
  # done
  log "[global] Global loop is commented out for safety. Set SINGLE_TILE=0 and uncomment to scale up."
fi

log "Pipeline complete."
