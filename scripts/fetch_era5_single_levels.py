"""
Download ERA5 single levels as NetCDF clipped to a bounding box and date range.

Variables:
- 100m_u_component_of_wind (u100)
- 100m_v_component_of_wind (v100)
- surface_pressure (sp)
- total_precipitation (tp)   # also available here; we’ll keep tp in this file too

This uses cdsapi's "reanalysis-era5-single-levels".
"""

import os, argparse, datetime as dt
import cdsapi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start", type=str, required=True)  # 2024-06-01T00:00
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--outfile", type=str, default="data/era5_single_levels.nc")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    c = cdsapi.Client()

    # Build lists of years/months/days/hours
    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)
    years = list({str(y) for y in range(t0.year, t1.year + 1)})
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = [f"{h:02d}:00" for h in range(24)]

    # NOTE: For small studies it’s common to pull a broader time window then subset locally
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",  # helpful for features
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "surface_pressure",
                "total_precipitation",
                "2m_temperature",
                "2m_dewpoint_temperature",  # for cross-check
            ],
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "area": [
                args.lat_max,
                args.lon_min,
                args.lat_min,
                args.lon_max,
            ],  # N, W, S, E
            "format": "netcdf",
        },
        args.outfile,
    )
    print("Saved", args.outfile)


if __name__ == "__main__":
    main()
