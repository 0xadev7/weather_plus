#!/usr/bin/env bash
set -euo pipefail

# Example: Iberia region + nearby Atlantic (tweak for your coverage or make multiple tiles globally)
LAT_MIN=20.0; LAT_MAX=60.0; LON_MIN=-30.0; LON_MAX=30.0
START=2025-09-15T00:00
END=2025-11-01T00:00

# 1) Baselines: OM + IFS (hourly), coarsely gridded to keep size reasonable
python weather_plus/scripts/fetch_openmeteo_hindcast.py \
--lat-min $LAT_MIN --lat-max $LAT_MAX --lon-min $LON_MIN --lon-max $LON_MAX \
--lat-steps 17 --lon-steps 25 \
--start $START --end $END --chunk-hours 168

# 2) Truth: ERA5 Single + ERA5-Land (hourly)
python weather_plus/scripts/fetch_era5_single_levels.py \
--lat-min $LAT_MIN --lat-max $LAT_MAX --lon-min $LON_MIN --lon-max $LON_MAX \
--start $START --end $END --outfile data/era5_single_levels.nc

python weather_plus/scripts/fetch_era5_land.py \
--lat-min $LAT_MIN --lat-max $LAT_MAX --lon-min $LON_MIN --lon-max $LON_MAX \
--start $START --end $END --outfile data/era5_land.nc

# 3) Pair into tidy Parquet with dual baselines
python weather_plus/scripts/make_training_pairs.py

# 4) Train all variables (dual-baseline aware)
python weather_plus/scripts/train_t2m.py
python weather_plus/scripts/train_td2m.py
python weather_plus/scripts/train_psfc.py
python weather_plus/scripts/train_wind_speed_100m.py
python weather_plus/scripts/train_wind_direction_100m.py
python weather_plus/scripts/train_tp.py

# 5) Verify: show gains vs OM and IFS
python weather_plus/scripts/verify_vs_openmeteo.py
