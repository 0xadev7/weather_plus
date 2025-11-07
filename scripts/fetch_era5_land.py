"""
Download ERA5-Land hourly reanalysis for t2m/d2m and tp for the same bbox/time.
Good for point-scale truth over land.

Dataset: reanalysis-era5-land
"""

import os, argparse, datetime as dt
import cdsapi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat-min", type=float, required=True)
    ap.add_argument("--lat-max", type=float, required=True)
    ap.add_argument("--lon-min", type=float, required=True)
    ap.add_argument("--lon-max", type=float, required=True)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--outfile", type=str, default="data/era5_land.nc")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    c = cdsapi.Client()

    t0 = dt.datetime.fromisoformat(args.start)
    t1 = dt.datetime.fromisoformat(args.end)
    years = list({str(y) for y in range(t0.year, t1.year + 1)})
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    hours = [f"{h:02d}" for h in range(24)]

    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "total_precipitation",
                "surface_pressure",
            ],
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "area": [args.lat_max, args.lon_min, args.lat_min, args.lon_max],
            "format": "netcdf",
        },
        args.outfile,
    )
    print("Saved", args.outfile)


if __name__ == "__main__":
    main()
