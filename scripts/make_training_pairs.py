#!/usr/bin/env python
import os, json, glob, datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

OM_DIR = os.path.join("data", "om_baseline")
OUT_DIR = os.path.join("data", "train")
os.makedirs(OUT_DIR, exist_ok=True)


def flatten_grid(lat, lon):
    return [(a, b) for a in lat for b in lon]


def to_celsius(k):
    return k - 273.15


def pa_to_hpa(pa):
    return pa / 100.0


def m_to_mm(m):
    return m * 1000.0


def ms_to_kmh(ms):
    return ms * 3.6


def nearest_idx(vals, x):
    vals = np.asarray(vals)
    return int(np.argmin(np.abs(vals - x)))


def pd_index_to_pylist(idx):
    return [ts.to_pydatetime().replace(tzinfo=None) for ts in idx.to_pydatetime()]


def load_chunk(path):
    with open(path, "r") as f:
        j = json.load(f)
    lat, lon = j["meta"]["lat"], j["meta"]["lon"]
    om = j["om"]["hourly"]
    ifs = j["ifs"]["hourly"] if j.get("ifs") else {}
    times = [
        dt.datetime.fromisoformat(t.replace("Z", "")) for t in j["om"]["hourly"]["time"]
    ]
    return lat, lon, times, om, ifs


def as_tg(arr, T, G):
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape == (T, G):
        return arr
    if arr.size == T * G:
        return arr.reshape(T, G)
    raise RuntimeError("wrong shape")


def main():
    # Truth files (ERA5 single + ERA5-Land)
    era5_single = (
        xr.open_dataset("data/era5_single_levels.nc")
        if os.path.exists("data/era5_single_levels.nc")
        else None
    )
    era5_land = (
        xr.open_dataset("data/era5_land.nc")
        if os.path.exists("data/era5_land.nc")
        else None
    )
    if era5_single is None and era5_land is None:
        raise SystemExit("Fetch ERA5 first.")

    rows = {"t2m": [], "td2m": [], "psfc": [], "tp": [], "wspd100": [], "wdir100": []}

    files = sorted(glob.glob(os.path.join(OM_DIR, "omifs_*.json")))
    if not files:
        raise SystemExit("No baseline files found. Run fetch_openmeteo_hindcast.py")

    for p in files:
        lat, lon, times, om, ifs = load_chunk(p)
        grid = flatten_grid(lat, lon)
        T, G = len(times), len(grid)

        # Leads/HOD
        t0 = times[0]
        leads = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
        hods = np.array([t.hour + t.minute / 60 for t in times])

        def base(name):
            om_arr = as_tg(om[name], T, G)
            ifs_arr = as_tg(ifs[name], T, G) if (ifs and name in ifs) else None
            return om_arr, ifs_arr

        # Baselines (OM + IFS) for all vars
        b_t2m_om, b_t2m_ifs = base("temperature_2m")
        b_td2m_om, b_td2m_ifs = base("dew_point_2m")
        b_psfc_om, b_psfc_ifs = base("surface_pressure")
        b_tp_om, b_tp_ifs = base("precipitation")
        b_wspd_om, b_wspd_ifs = base("wind_speed_100m")
        b_wdir_om, b_wdir_ifs = base("wind_direction_100m")

        # Truth from ERA5/ERA5-Land
        def nearest_truth(ds, var, grid, times):
            lats = ds.coords["latitude"].values
            lons = ds.coords["longitude"].values
            tlist = pd_index_to_pylist(ds.indexes["time"])
            out = np.empty((T, G), dtype=float)
            for gi, (la, lo) in enumerate(grid):
                ilat = nearest_idx(lats, la)
                ilon = nearest_idx(lons, lo)
                for ti, tt in enumerate(times):
                    it = nearest_idx(tlist, tt)
                    out[ti, gi] = float(
                        ds[var].isel(latitude=ilat, longitude=ilon, time=it).values
                    )
            return out

        # choose ERA5-Land for t2m/td2m/tp if available
        if era5_land is not None:
            t2m_truth = to_celsius(nearest_truth(era5_land, "t2m", grid, times))
            td2m_truth = to_celsius(nearest_truth(era5_land, "d2m", grid, times))
            tp_truth = m_to_mm(nearest_truth(era5_land, "tp", grid, times))
            sp_truth = (
                pa_to_hpa(nearest_truth(era5_land, "sp", grid, times))
                if "sp" in era5_land.variables
                else None
            )
        else:
            t2m_truth = to_celsius(nearest_truth(era5_single, "t2m", grid, times))
            td2m_truth = to_celsius(nearest_truth(era5_single, "d2m", grid, times))
            tp_truth = m_to_mm(nearest_truth(era5_single, "tp", grid, times))
            sp_truth = pa_to_hpa(nearest_truth(era5_single, "sp", grid, times))

        if sp_truth is None:
            sp_truth = pa_to_hpa(nearest_truth(era5_single, "sp", grid, times))

        u100 = nearest_truth(era5_single, "u100", grid, times)
        v100 = nearest_truth(era5_single, "v100", grid, times)
        wspd_truth = ms_to_kmh(np.hypot(u100, v100))
        wdir_truth = (np.degrees(np.arctan2(-u100, -v100)) + 360.0) % 360.0

        def emit(rows_list, base_om, base_ifs, truth):
            for gi, (la, lo) in enumerate(grid):
                for ti, tt in enumerate(times):
                    bom = float(base_om[ti, gi])
                    bifs = float(base_ifs[ti, gi]) if base_ifs is not None else np.nan
                    rows_list.append(
                        {
                            "lat": la,
                            "lon": lo,
                            "hod": hods[ti],
                            "lead": leads[ti],
                            "baseline_om": bom,
                            "baseline_ifs": bifs,
                            "baseline_diff": (
                                bom - (bifs if np.isfinite(bifs) else bom)
                            ),
                            "target": float(truth[ti, gi]),
                            "time": tt.isoformat(),
                        }
                    )

        emit(rows["t2m"], b_t2m_om, b_t2m_ifs, t2m_truth)
        emit(rows["td2m"], b_td2m_om, b_td2m_ifs, td2m_truth)
        emit(rows["psfc"], b_psfc_om, b_psfc_ifs, sp_truth)
        emit(rows["tp"], b_tp_om, b_tp_ifs, tp_truth)
        emit(rows["wspd100"], b_wspd_om, b_wspd_ifs, wspd_truth)
        emit(rows["wdir100"], b_wdir_om, b_wdir_ifs, wdir_truth)

        print("paired", os.path.basename(p))

    pd.DataFrame(rows["t2m"]).to_parquet(os.path.join(OUT_DIR, "t2m_training.parquet"))
    pd.DataFrame(rows["td2m"]).to_parquet(
        os.path.join(OUT_DIR, "td2m_training.parquet")
    )
    pd.DataFrame(rows["psfc"]).to_parquet(
        os.path.join(OUT_DIR, "psfc_training.parquet")
    )
    pd.DataFrame(rows["tp"]).to_parquet(os.path.join(OUT_DIR, "tp_training.parquet"))
    pd.DataFrame(rows["wspd100"]).to_parquet(
        os.path.join(OUT_DIR, "wspd100_training.parquet")
    )
    pd.DataFrame(rows["wdir100"]).to_parquet(
        os.path.join(OUT_DIR, "wdir100_training.parquet")
    )
    print("Saved parquet tables in ./data/train/")


if __name__ == "__main__":
    main()
