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
    raise RuntimeError(
        f"wrong shape: expected {(T,G)}, got {arr.shape} from size {arr.size}"
    )


def _get_lat_lon_lists(j):
    """
    Try many schema variants for grid lat/lon arrays.
    Returns (lat_list, lon_list) or (None, None) if not found or invalid.
    """
    candidates = [
        ("meta", "lat"),
        ("meta", "lon"),
        ("meta", "lats"),
        ("meta", "lons"),
        ("meta", "latitude"),
        ("meta", "longitude"),
        ("meta.grid", "lat"),
        ("meta.grid", "lon"),
        ("coords", "lat"),
        ("coords", "lon"),
        ("om", "latitude"),
        ("om", "longitude"),
        ("om.meta", "lat"),
        ("om.meta", "lon"),
        ("om.grid", "lat"),
        ("om.grid", "lon"),
    ]

    def dig(root, path):
        cur = root
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    # We need a pair, so probe pairs in order
    pairs = [
        (("meta", "lat"), ("meta", "lon")),
        (("meta", "lats"), ("meta", "lons")),
        (("meta", "latitude"), ("meta", "longitude")),
        (("meta.grid", "lat"), ("meta.grid", "lon")),
        (("coords", "lat"), ("coords", "lon")),
        (("om", "latitude"), ("om", "longitude")),
        (("om.meta", "lat"), ("om.meta", "lon")),
        (("om.grid", "lat"), ("om.grid", "lon")),
    ]

    for (p_lat, k_lat), (p_lon, k_lon) in pairs:
        lat = dig(j, p_lat)
        if isinstance(lat, dict):  # e.g. {"lat":[...]}
            lat = lat.get(k_lat)
        elif p_lat.count("."):  # nested handled above
            lat = dig(j, p_lat.split(".")[0])  # fallback noop

        lat = dig(j, p_lat) if isinstance(lat, type(None)) else lat
        if isinstance(lat, dict):
            lat = lat.get(k_lat)

        # Now lon
        lon = dig(j, p_lon)
        if isinstance(lon, dict):
            lon = lon.get(k_lon)

        # If we still don't have arrays, try direct keys
        if lat is None:
            lat = dig(j, p_lat + "." + k_lat)
        if lon is None:
            lon = dig(j, p_lon + "." + k_lon)

        # Validate
        if lat is not None and lon is not None:
            # wrap scalars
            if not isinstance(lat, (list, tuple, np.ndarray)):
                lat = [lat]
            if not isinstance(lon, (list, tuple, np.ndarray)):
                lon = [lon]
            lat = [float(x) for x in lat]
            lon = [float(x) for x in lon]
            if len(lat) >= 1 and len(lon) >= 1:
                return lat, lon

    return None, None


def _safe_get(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


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
        try:
            with open(p, "r") as f:
                j = json.load(f)
        except Exception as e:
            print(f"[pair] skip {os.path.basename(p)} (invalid JSON: {e})")
            continue

        # Pull grid
        lat, lon = None, None
        meta = j.get("meta", {})
        # First try declared helper
        lat, lon = _get_lat_lon_lists(j)

        if lat is None or lon is None:
            print(
                f"[pair] skip {os.path.basename(p)} (no lat/lon grid in meta/om/coords)"
            )
            continue

        # Pull hourly block(s)
        om = _safe_get(j, "om", "hourly", default=None)
        if om is None or "time" not in om:
            print(f"[pair] skip {os.path.basename(p)} (missing om.hourly.time)")
            continue
        ifs = _safe_get(j, "ifs", "hourly", default=None)

        # Times
        try:
            times = [dt.datetime.fromisoformat(t.replace("Z", "")) for t in om["time"]]
        except Exception as e:
            print(f"[pair] skip {os.path.basename(p)} (bad time array: {e})")
            continue

        grid = flatten_grid(lat, lon)
        T, G = len(times), len(grid)
        if T == 0 or G == 0:
            print(f"[pair] skip {os.path.basename(p)} (empty times or grid)")
            continue

        t0 = times[0]
        leads = np.array(
            [(t - t0).total_seconds() / 3600.0 for t in times], dtype=float
        )
        hods = np.array([t.hour + t.minute / 60 for t in times], dtype=float)

        # Baseline helpers
        def base(name):
            # OM may store arrays per location in flattened order matching grid
            if name not in om:
                raise KeyError(name)
            om_arr = as_tg(om[name], T, G)
            ifs_arr = None
            if ifs is not None and name in ifs:
                ifs_arr = as_tg(ifs[name], T, G)
            return om_arr, ifs_arr

        try:
            b_t2m_om, b_t2m_ifs = base("temperature_2m")
            b_td2m_om, b_td2m_ifs = base("dew_point_2m")
            b_psfc_om, b_psfc_ifs = base("surface_pressure")
            b_tp_om, b_tp_ifs = base("precipitation")
            b_wspd_om, b_wspd_ifs = base("wind_speed_100m")
            b_wdir_om, b_wdir_ifs = base("wind_direction_100m")
        except KeyError as e:
            print(f"[pair] skip {os.path.basename(p)} (missing baseline key: {e})")
            continue
        except RuntimeError as e:
            print(f"[pair] skip {os.path.basename(p)} (shape issue: {e})")
            continue

        # Truth samplers
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

        def prefer_land(var):
            return era5_land if var in era5_land.variables else era5_single

        try:
            t2m_truth = to_celsius(
                nearest_truth(prefer_land("t2m"), "t2m", grid, times)
            )
            td2m_truth = to_celsius(
                nearest_truth(prefer_land("d2m"), "d2m", grid, times)
            )
            tp_truth = m_to_mm(nearest_truth(prefer_land("tp"), "tp", grid, times))
            if "sp" in prefer_land("sp").variables:
                sp_truth = pa_to_hpa(
                    nearest_truth(prefer_land("sp"), "sp", grid, times)
                )
            else:
                sp_truth = pa_to_hpa(nearest_truth(era5_single, "sp", grid, times))

            u100 = nearest_truth(era5_single, "u100", grid, times)
            v100 = nearest_truth(era5_single, "v100", grid, times)
            wspd_truth = ms_to_kmh(np.hypot(u100, v100))
            wdir_truth = (np.degrees(np.arctan2(-u100, -v100)) + 360.0) % 360.0
        except Exception as e:
            print(f"[pair] skip {os.path.basename(p)} (ERA5 sample error: {e})")
            continue

        def emit(rows_list, bom, bifs, truth):
            for gi, (la, lo) in enumerate(grid):
                for ti, tt in enumerate(times):
                    bomv = float(bom[ti, gi])
                    bifsv = float(bifs[ti, gi]) if bifs is not None else np.nan
                    rows_list.append(
                        {
                            "lat": la,
                            "lon": lo,
                            "hod": float(hods[ti]),
                            "lead": float(leads[ti]),
                            "baseline_om": bomv,
                            "baseline_ifs": bifsv,
                            "baseline_diff": bomv
                            - (bifsv if np.isfinite(bifsv) else bomv),
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
        else:
            print(
                f"[pair] warning: empty dataframe for {outname} on {args.tile_id} (no matching OM files / coords?)"
            )


if __name__ == "__main__":
    main()
