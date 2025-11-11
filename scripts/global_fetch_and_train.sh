#!/usr/bin/env bash
# Global fetch + train pipeline (resume-friendly, OM-free-plan safe)
# - Batches + throttles Open-Meteo via env passed to the Python fetcher
# - Skips work that already exists (idempotent)
# - Lets you filter which tiles to run (by index or regex)
# - Has a dry-run mode and clear logging
#
# Usage:
#   scripts/global_fetch_and_train.sh
#
# Optional env overrides:
#   START=2025-09-01T00:00  END=2025-11-01T00:00
#   OM_MAX_LOCS=25 OM_RPS=0.33 OM_RETRIES_429=6 OM_TIMEOUT=180
#   LAT_STEPS=7 LON_STEPS=9  CHUNK_HOURS=168
#   TILE_INCLUDE_REGEX="LAT(2|3)_LON.*"   TILE_EXCLUDE_REGEX="LAT0_.*"
#   DRY_RUN=1 ERA5_ONLY=1 OM_ONLY=1 CLEAN_INTERMEDIATES=1
#   OM_FETCH_SCRIPT="scripts/fetch_openmeteo_hindcast.py"
#   ERA5_SINGLE_SCRIPT="scripts/fetch_era5_single_levels.py"   # or era5_fetcher_zip_aware.py
#   ERA5_LAND_SCRIPT="scripts/fetch_era5_land.py"              # or era5_land_zip_aware.py
set -euo pipefail

export HDF5_USE_FILE_LOCKING=FALSE

# -----------------------------
# Config (env overridable)
# -----------------------------
START="${START:-2025-10-01T00:00}"
END="${END:-2025-11-01T00:00}"

LAT_STEPS="${LAT_STEPS:-7}"
LON_STEPS="${LON_STEPS:-9}"
CHUNK_HOURS="${CHUNK_HOURS:-168}"

# OM free plan knobs (forwarded to Python)
export OM_MAX_LOCS="${OM_MAX_LOCS:-25}"
export OM_RPS="${OM_RPS:-0.33}"
export OM_RETRIES_429="${OM_RETRIES_429:-6}"
export OM_TIMEOUT="${OM_TIMEOUT:-180}"

# Script paths (override if you renamed to the ZIP-aware versions)
OM_FETCH_SCRIPT="${OM_FETCH_SCRIPT:-scripts/fetch_openmeteo_hindcast.py}"
ERA5_SINGLE_SCRIPT="${ERA5_SINGLE_SCRIPT:-scripts/fetch_era5_single_levels.py}"
ERA5_LAND_SCRIPT="${ERA5_LAND_SCRIPT:-scripts/fetch_era5_land.py}"

# Tile filters / modes
TILE_INCLUDE_REGEX="${TILE_INCLUDE_REGEX:-}"  # empty = include all
TILE_EXCLUDE_REGEX="${TILE_EXCLUDE_REGEX:-}"  # empty = exclude none
DRY_RUN="${DRY_RUN:-0}"
ERA5_ONLY="${ERA5_ONLY:-0}"
OM_ONLY="${OM_ONLY:-0}"
CLEAN_INTERMEDIATES="${CLEAN_INTERMEDIATES:-0}"

# 30° bands (coarse global)
LAT_EDGES=("-90" "-60" "-30" "0" "30" "60" "90")
LON_EDGES=("-180" "-150" "-120" "-90" "-60" "-30" "0" "30" "60" "90" "120" "150" "180")

OUT_OM_DIR="data/om_baseline"
OUT_TILE_DIR="tiles"
mkdir -p "$OUT_OM_DIR" "$OUT_TILE_DIR" logs

# -----------------------------
# Helpers
# -----------------------------
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# Plain runner (no tee)
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] $*"
  else
    eval "$@"
  fi
}

# Runner that logs to a file with tee, but still handles DRY_RUN correctly
run_logged() {
  local cmd="$1"
  local logfile="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] $cmd"
    # Write a dry-run marker into the log so it’s obvious what would have run
    echo "[$(ts)] [dry-run] $cmd" >> "$logfile"
  else
    # Use bash -lc so we can pass a full pipeline safely
    bash -lc "$cmd 2>&1 | tee -a \"$logfile\""
  fi
}

should_run_tile() {
  local tile="$1"
  if [[ -n "$TILE_INCLUDE_REGEX" && ! "$tile" =~ $TILE_INCLUDE_REGEX ]]; then
    return 1
  fi
  if [[ -n "$TILE_EXCLUDE_REGEX" && "$tile" =~ $TILE_EXCLUDE_REGEX ]]; then
    return 1
  fi
  return 0
}

# Presence checks to skip work
have_era5_files() {
  local tile="$1"
  [[ -s "${OUT_TILE_DIR}/${tile}_era5_single.nc" && -s "${OUT_TILE_DIR}/${tile}_era5_land.nc" ]]
}

have_pairs_done() {
  local tile="$1"
  [[ -s "${OUT_TILE_DIR}/${tile}.pairs.done" ]]
}

have_om_any_batch_for_timeslice() {
  # Check at least one omifs_* file for a time slice exists (lenient resume)
  local start_tag="$1"  # YYYYmmddHH
  local end_tag="$2"    # YYYYmmddHH
  compgen -G "${OUT_OM_DIR}/omifs_${start_tag}_${end_tag}_b*_n*.json" > /dev/null
}

# Convert ISO to tag used by fetcher (YYYYmmddHH)
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

# Compute end tag for the first chunk (START + CHUNK_HOURS)
first_chunk_end_tag() {
  python - "$START" "$CHUNK_HOURS" <<'PY'
import sys, datetime as dt
s=sys.argv[1]; ch=int(sys.argv[2])
t0=dt.datetime.fromisoformat(s)
t1=t0+dt.timedelta(hours=ch)
print(t1.strftime("%Y%m%d%H"))
PY
}

# -----------------------------
# Main
# -----------------------------
log "Global fetch/train starting: START=$START END=$END LAT_STEPS=$LAT_STEPS LON_STEPS=$LON_STEPS CHUNK_HOURS=$CHUNK_HOURS"
log "OM knobs: OM_MAX_LOCS=$OM_MAX_LOCS OM_RPS=$OM_RPS OM_RETRIES_429=$OM_RETRIES_429 OM_TIMEOUT=$OM_TIMEOUT"
[[ "$DRY_RUN" == "1" ]] && log "DRY_RUN=1 (no commands will execute)"
[[ "$ERA5_ONLY" == "1" ]] && log "Mode: ERA5_ONLY (skip OM)"
[[ "$OM_ONLY" == "1" ]] && log "Mode: OM_ONLY (skip ERA5 and pairing)"

trap 'log "Interrupted"; exit 130' INT

S_TAG="$(iso_to_tag "$START")"
E_TAG="$(first_chunk_end_tag)"

for ((i=0; i<${#LAT_EDGES[@]}-1; i++)); do
  for ((j=0; j<${#LON_EDGES[@]}-1; j++)); do
    LAT_MIN=${LAT_EDGES[$i]}
    LAT_MAX=${LAT_EDGES[$i+1]}
    LON_MIN=${LON_EDGES[$j]}
    LON_MAX=${LON_EDGES[$j+1]}
    TILE="LAT${i}_LON${j}"

    if ! should_run_tile "$TILE"; then
      log "[skip] $TILE (filtered)"
      continue
    fi

    log "=== Tile $TILE (${LAT_MIN},${LAT_MAX},${LON_MIN},${LON_MAX}) ==="

    # 1) Open-Meteo (batched & throttled by the Python script)
    if [[ "$ERA5_ONLY" != "1" ]]; then
      if have_om_any_batch_for_timeslice "$S_TAG" "$E_TAG"; then
        log "[om] some batches already present for first slice (${S_TAG}_${E_TAG}), will still run to fill gaps."
      fi

      log "[om] Fetching OM baselines for $TILE"
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

    # 2) ERA5 (two files) per tile
    if [[ "$OM_ONLY" != "1" ]]; then
      if have_era5_files "$TILE"; then
        log "[era5] already fetched for $TILE"
      else
        log "[era5] Fetching ERA5 single levels for $TILE"
        run_logged \
          "python \"$ERA5_SINGLE_SCRIPT\" \
            --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
            --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
            --start \"$START\" --end \"$END\" \
            --debug-merge --prune-parts \
            --outfile \"${OUT_TILE_DIR}/${TILE}_era5_single.nc\"" \
          "logs/${TILE}_era5_single.log"

        log "[era5] Fetching ERA5 land for $TILE"
        run_logged \
          "python \"$ERA5_LAND_SCRIPT\" \
            --lat-min \"$LAT_MIN\" --lat-max \"$LAT_MAX\" \
            --lon-min \"$LON_MIN\" --lon-max \"$LON_MAX\" \
            --start \"$START\" --end \"$END\" \
            --debug-merge --prune-parts \
            --outfile \"${OUT_TILE_DIR}/${TILE}_era5_land.nc\"" \
          "logs/${TILE}_era5_land.log"
      fi

      # 3) Pairing
      if have_pairs_done "$TILE"; then
        log "[pair] already paired for $TILE"
      else
        log "[pair] Making training pairs for $TILE"
        run_logged \
          "python scripts/make_training_pairs_tile.py \
            --era5-single \"${OUT_TILE_DIR}/${TILE}_era5_single.nc\" \
            --era5-land   \"${OUT_TILE_DIR}/${TILE}_era5_land.nc\" \
            --tile-id     \"$TILE\"" \
          "logs/${TILE}_pair.log"

        if [[ "$DRY_RUN" != "1" ]]; then
          : > "${OUT_TILE_DIR}/${TILE}.pairs.done"
        fi

        if [[ "$CLEAN_INTERMEDIATES" == "1" && "$DRY_RUN" != "1" ]]; then
          log "[clean] removing ERA5 intermediates for $TILE"
          rm -f "${OUT_TILE_DIR}/${TILE}_era5_single.nc" "${OUT_TILE_DIR}/${TILE}_era5_land.nc" || true
        fi
      fi
    else
      log "[era5/pair] skipped (OM_ONLY=1)"
    fi

    log "=== Done $TILE ==="
  done
done

log "All tiles processed."
