#!/usr/bin/env bash
set -euo pipefail

# Choose coarse global tiles (30° bands). You can refine where needed.
LAT_EDGES=("-90" "-60" "-30" "0" "30" "60" "90")
LON_EDGES=("-180" "-150" "-120" "-90" "-60" "-30" "0" "30" "60" "90" "120" "150" "180")

START=2025-09-01T00:00
END=2025-11-01T00:00

mkdir -p data/om_baseline tiles

for ((i=0; i<${#LAT_EDGES[@]}-1; i++)); do
    for ((j=0; j<${#LON_EDGES[@]}-1; j++)); do
        LAT_MIN=${LAT_EDGES[$i]}
        LAT_MAX=${LAT_EDGES[$i+1]}
        LON_MIN=${LON_EDGES[$j]}
        LON_MAX=${LON_EDGES[$j+1]}
        
        # Skip near-empty small ocean-only tiles if you want, or keep for uniformity.
        TILE="LAT${i}_LON${j}"
        
        echo "=== Fetching baselines for $TILE ($LAT_MIN,$LAT_MAX,$LON_MIN,$LON_MAX)"
        python weather_plus/scripts/fetch_openmeteo_hindcast.py \
        --lat-min $LAT_MIN --lat-max $LAT_MAX --lon-min $LON_MIN --lon-max $LON_MAX \
        --lat-steps 7 --lon-steps 9 \
        --start $START --end $END --chunk-hours 168
        
        echo "=== Fetching ERA5 for $TILE"
        python weather_plus/scripts/fetch_era5_single_levels.py \
        --lat-min $LAT_MIN --lat-max $LAT_MAX --lon-min $LON_MIN --lon-max $LON_MAX \
        --start $START --end $END --outfile tiles/${TILE}_era5_single.nc
        
        python weather_plus/scripts/fetch_era5_land.py \
        --lat-min $LAT_MIN --lat-max $LAT_MAX --lon-min $LON_MIN --lon-max $LON_MAX \
        --start $START --end $END --outfile tiles/${TILE}_era5_land.nc
        
        echo "=== Pairing for $TILE"
        python weather_plus/scripts/make_training_pairs_tile.py \
        --era5-single tiles/${TILE}_era5_single.nc \
        --era5-land   tiles/${TILE}_era5_land.nc \
        --tile-id     $TILE
    done
done
