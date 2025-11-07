#!/usr/bin/env python
import os, json, glob, argparse, datetime as dt
import numpy as np, pandas as pd, xarray as xr

OM_DIR = os.path.join("data", "om_baseline")
OUT_DIR = os.path.join("data", "train_tiles")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-single", required=True)
    ap.add_argument("--era5-land", required=True)
    ap.add_argument("--tile-id", required=True)
    args = ap.parse_args()

    era5_single = xr.open_dataset(args.era5_single)
    era5_land = xr.open_dataset(args.era5_land)

    rows = {k: [] for k in ["t2m", "td2m", "psfc", "tp", "wspd100", "wdir100"]}

    files = sorted(glob.glob(os.path.join(OM_DIR, "omifs_*.json")))
    if not files:
        raise SystemExit("No OM baseline files found")

    for p in files:
        with open(p, "r") as f:
            j = json.load(f)
        lat, lon = j["meta"]["lat"], j["meta"]["lon"]
        # Skip files outside this tile’s bbox quickly
        if max(lat) < float(era5_single.latitude.min()) or min(lat) > float(
            era5_single.latitude.max()
        ):
            continue
        if max(lon) < float(era5_single.longitude.min()) or min(lon) > float(
            era5_single.longitude.max()
        ):
            continue

        times = [
            dt.datetime.fromisoformat(t.replace("Z", ""))
            for t in j["om"]["hourly"]["time"]
        ]
        om, ifs = j["om"]["hourly"], (j["ifs"]["hourly"] if j.get("ifs") else {})
        grid = flatten_grid(lat, lon)
        T, G = len(times), len(grid)

        t0 = times[0]
        leads = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
        hods = np.array([t.hour + t.minute / 60 for t in times])

        def base(name):
            om_arr = as_tg(om[name], T, G)
            ifs_arr = as_tg(ifs[name], T, G) if (ifs and name in ifs) else None
            return om_arr, ifs_arr

        b_t2m_om, b_t2m_ifs = base("temperature_2m")
        b_td2m_om, b_td2m_ifs = base("dew_point_2m")
        b_psfc_om, b_psfc_ifs = base("surface_pressure")
        b_tp_om, b_tp_ifs = base("precipitation")
        b_wspd_om, b_wspd_ifs = base("wind_speed_100m")
        b_wdir_om, b_wdir_ifs = base("wind_direction_100m")

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

        # Land where valid; else fallback to single levels
        def prefer_land(var):
            if var in era5_land.variables:
                return era5_land
            return era5_single

        t2m_truth = to_celsius(nearest_truth(prefer_land("t2m"), "t2m", grid, times))
        td2m_truth = to_celsius(nearest_truth(prefer_land("d2m"), "d2m", grid, times))
        tp_truth = m_to_mm(nearest_truth(prefer_land("tp"), "tp", grid, times))
        sp_truth = (
            pa_to_hpa(nearest_truth(prefer_land("sp"), "sp", grid, times))
            if "sp" in prefer_land("sp").variables
            else pa_to_hpa(nearest_truth(era5_single, "sp", grid, times))
        )

        u100 = nearest_truth(era5_single, "u100", grid, times)
        v100 = nearest_truth(era5_single, "v100", grid, times)
        wspd_truth = ms_to_kmh(np.hypot(u100, v100))
        wdir_truth = (np.degrees(np.arctan2(-u100, -v100)) + 360.0) % 360.0

        def emit(rows_list, bom, bifs, truth):
            for gi, (la, lo) in enumerate(grid):
                for ti, tt in enumerate(times):
                    bomv = float(bom[ti, gi])
                    bifsv = float(bifs[ti, gi]) if bifs is not None else np.nan
                    rows_list.append(
                        {
                            "lat": la,
                            "lon": lo,
                            "hod": hods[ti],
                            "lead": leads[ti],
                            "baseline_om": bomv,
                            "baseline_ifs": bifsv,
                            "baseline_diff": (
                                bomv - (bifsv if np.isfinite(bifsv) else bomv)
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

        print("paired", os.path.basename(p), "→", args.tile_id)

    # Write per-tile parquet
    for key, outname in [
        ("t2m", "t2m"),
        ("td2m", "td2m"),
        ("psfc", "psfc"),
        ("tp", "tp"),
        ("wspd100", "wspd100"),
        ("wdir100", "wdir100"),
    ]:
        df = pd.DataFrame(rows[key])
        if not df.empty:
            df.to_parquet(os.path.join(OUT_DIR, f"{args.tile_id}__{outname}.parquet"))
            print("saved", args.tile_id, outname)


if __name__ == "__main__":
    main()
