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
    Return (lat_list, lon_list) as floats, handling many schemas:

    1) meta.lat / meta.lon : [floats] or [objects]
    2) meta.lats / meta.lons
    3) meta.grid.lat / meta.grid.lon
    4) coords.lat / coords.lon
    5) om.latitude / om.longitude
    6) om.meta.lat / om.meta.lon
    7) om.grid.lat / om.grid.lon
    8) A single list of points: *.grid.points = [{lat:.., lon:..}, ...] or *.points = [...]
    9) A flat list of pairs: [[lat, lon], ...] or [{"lat":..,"lon":..}, ...]
    """

    def dig(root, dotted):
        cur = root
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _to_float_seq(seq, key_candidates=("lat", "latitude", "y", "value")):
        """Coerce a heterogeneous list into a list[float]. Accept numbers, strings,
        dicts with key_candidates, or 1-2 element lists/tuples."""
        out = []
        for x in seq:
            if isinstance(x, (int, float)):
                out.append(float(x))
                continue
            if isinstance(x, str):
                try:
                    out.append(float(x))
                    continue
                except:
                    pass
            if isinstance(x, dict):
                got = None
                for kk in key_candidates:
                    if kk in x:
                        vv = x[kk]
                        if isinstance(vv, (int, float)):
                            got = float(vv)
                            break
                        if isinstance(vv, str):
                            try:
                                got = float(vv)
                                break
                            except:
                                pass
                if got is not None:
                    out.append(got)
                    continue
            if isinstance(x, (list, tuple)) and len(x) >= 1:
                v = x[0]
                if isinstance(v, (int, float)):
                    out.append(float(v))
                    continue
                if isinstance(v, str):
                    try:
                        out.append(float(v))
                        continue
                    except:
                        pass
        return out if out else None

    def _from_points(seq):
        """seq like [{'lat':..,'lon':..}, ...] or [[lat, lon], ...] -> unique sorted lat[], lon[]."""
        lats, lons = [], []
        for p in seq:
            la = lo = None
            if isinstance(p, dict):
                for k in ("lat", "latitude", "y"):
                    if k in p:
                        v = p[k]
                        la = float(v) if isinstance(v, (int, float)) else float(str(v))
                        break
                for k in ("lon", "longitude", "x"):
                    if k in p:
                        v = p[k]
                        lo = float(v) if isinstance(v, (int, float)) else float(str(v))
                        break
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    la = float(p[0])
                    lo = float(p[1])
                except:
                    pass
            if la is not None and lo is not None:
                lats.append(la)
                lons.append(lo)
        if lats and lons:
            # dedupe while preserving order-ish
            lat_u = sorted(set(round(v, 6) for v in lats))
            lon_u = sorted(set(round(v, 6) for v in lons))
            return [float(v) for v in lat_u], [float(v) for v in lon_u]
        return None, None

    # Candidate pairs of arrays (lat[], lon[])
    array_pairs = [
        ("meta.lat", "meta.lon"),
        ("meta.lats", "meta.lons"),
        ("meta.latitude", "meta.longitude"),
        ("meta.grid.lat", "meta.grid.lon"),
        ("coords.lat", "coords.lon"),
        ("om.latitude", "om.longitude"),
        ("om.meta.lat", "om.meta.lon"),
        ("om.grid.lat", "om.grid.lon"),
    ]
    for p_lat, p_lon in array_pairs:
        lat_raw, lon_raw = dig(j, p_lat), dig(j, p_lon)
        if (
            lat_raw is not None
            and lon_raw is not None
            and isinstance(lat_raw, (list, tuple))
            and isinstance(lon_raw, (list, tuple))
        ):
            lat = _to_float_seq(lat_raw)
            lon = _to_float_seq(
                lon_raw, key_candidates=("lon", "longitude", "x", "value")
            )
            if lat and lon:
                return lat, lon

    # Points-style schemas
    point_candidates = [
        "meta.grid.points",
        "meta.points",
        "om.grid.points",
        "om.points",
        "coords.points",
    ]
    for path in point_candidates:
        seq = dig(j, path)
        if isinstance(seq, (list, tuple)) and seq:
            lat, lon = _from_points(seq)
            if lat and lon:
                return lat, lon

    # Sometimes everything is under a top-level 'grid'
    grid = j.get("grid")
    if isinstance(grid, dict):
        lat_raw, lon_raw = grid.get("lat"), grid.get("lon")
        if isinstance(lat_raw, (list, tuple)) and isinstance(lon_raw, (list, tuple)):
            lat = _to_float_seq(lat_raw)
            lon = _to_float_seq(
                lon_raw, key_candidates=("lon", "longitude", "x", "value")
            )
            if lat and lon:
                return lat, lon
        pts = grid.get("points")
        if isinstance(pts, (list, tuple)):
            lat, lon = _from_points(pts)
            if lat and lon:
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
            print(f"[pair] skip {os.path.basename(p)} (unable to parse lat/lon grid)")
            continue

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
