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


def _safe_get(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------
# Robust lat/lon parsing
# -------------------------
def _get_lat_lon_lists(j):
    """
    Return (lat_list, lon_list) as floats, handling many schemas (arrays, points,
    descriptors, bbox+resolution). This is the robust version that tolerates
    dict descriptors and mixed types.
    """

    def dig(root, dotted):
        cur = root
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    def _as_float(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
        raise ValueError

    def _ensure_sorted_unique(vals):
        return [float(v) for v in sorted(set(round(float(v), 7) for v in vals))]

    def _from_descriptor(desc, axis="lat"):
        if not isinstance(desc, dict):
            return None
        if "values" in desc and isinstance(desc["values"], (list, tuple)):
            out = []
            for v in desc["values"]:
                try:
                    out.append(_as_float(v))
                except Exception:
                    pass
            return out if out else None
        for a, b, n in (("start", "stop", "num"), ("min", "max", "n")):
            if a in desc and b in desc and n in desc:
                try:
                    a_v, b_v = _as_float(desc[a]), _as_float(desc[b])
                    n_v = int(desc[n])
                    if n_v <= 0:
                        return None
                    return list(np.linspace(a_v, b_v, n_v))
                except Exception:
                    pass
        if all(k in desc for k in ("start", "step", "count")):
            try:
                start = _as_float(desc["start"])
                step = _as_float(desc["step"])
                cnt = int(desc["count"])
                if cnt <= 0:
                    return None
                return [start + i * step for i in range(cnt)]
            except Exception:
                pass
        for k in ("array", "data"):
            if k in desc and isinstance(desc[k], (list, tuple)):
                try:
                    return [_as_float(v) for v in desc[k]]
                except Exception:
                    pass
        return None

    def _from_bbox(meta):
        if not isinstance(meta, dict):
            return None, None
        bbox = meta.get("bbox")
        if bbox is None:
            return None, None

        def _parse_bbox(bb):
            if isinstance(bb, dict):
                if all(k in bb for k in ("south", "north", "west", "east")):
                    return (
                        float(bb["south"]),
                        float(bb["north"]),
                        float(bb["west"]),
                        float(bb["east"]),
                    )
                if all(k in bb for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
                    return (
                        float(bb["lat_min"]),
                        float(bb["lat_max"]),
                        float(bb["lon_min"]),
                        float(bb["lon_max"]),
                    )
            elif isinstance(bb, (list, tuple)) and len(bb) == 4:
                return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            return None

        parsed = _parse_bbox(bbox)
        if not parsed:
            return None, None
        lat_min, lat_max, lon_min, lon_max = parsed

        nx = meta.get("nx")
        ny = meta.get("ny")
        n = meta.get("n")
        try:
            if nx is None and ny is None and n is not None:
                nx = ny = int(n)
            if nx is None or ny is None:
                res = meta.get("res") or meta.get("resolution")
                if res is not None:
                    nx = ny = int(res)
            nx = int(nx) if nx is not None else None
            ny = int(ny) if ny is not None else None
        except Exception:
            nx = ny = None
        if not nx or not ny:
            return None, None

        lat_list = list(np.linspace(lat_min, lat_max, ny))
        lon_list = list(np.linspace(lon_min, lon_max, nx))
        return lat_list, lon_list

    def _to_float_seq(seq, key_candidates=("lat", "latitude", "y", "value")):
        out = []
        for x in seq:
            try:
                out.append(_as_float(x))
                continue
            except Exception:
                pass
            if isinstance(x, dict):
                vals = _from_descriptor(x)
                if vals:
                    out.extend(vals)
                    continue
                got = None
                for kk in key_candidates:
                    if kk in x:
                        try:
                            got = _as_float(x[kk])
                            break
                        except Exception:
                            pass
                if got is not None:
                    out.append(got)
                    continue
            if isinstance(x, (list, tuple)) and len(x) >= 1:
                try:
                    out.append(_as_float(x[0]))
                    continue
                except Exception:
                    pass
        return out if out else None

    def _from_points(seq):
        lats, lons = [], []
        for p in seq:
            la = lo = None
            if isinstance(p, dict):
                for k in ("lat", "latitude", "y"):
                    if k in p:
                        try:
                            la = float(p[k])
                            break
                        except Exception:
                            pass
                for k in ("lon", "longitude", "x"):
                    if k in p:
                        try:
                            lo = float(p[k])
                            break
                        except Exception:
                            pass
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    la, lo = float(p[0]), float(p[1])
                except Exception:
                    pass
            if la is not None and lo is not None:
                lats.append(la)
                lons.append(lo)
        if lats and lons:
            return _ensure_sorted_unique(lats), _ensure_sorted_unique(lons)
        return None, None

    array_pairs = [
        ("meta.lat", "meta.lon"),
        ("meta.lats", "meta.lons"),
        ("meta.latitude", "meta.longitude"),
        ("meta.grid.lat", "meta.grid.lon"),
        ("coords.lat", "coords.lon"),
        ("om.latitude", "om.longitude"),
        ("om.meta.lat", "om.meta.lon"),
        ("om.grid.lat", "om.grid.lon"),
        ("grid.lat", "grid.lon"),
    ]
    for p_lat, p_lon in array_pairs:
        lat_raw, lon_raw = dig(j, p_lat), dig(j, p_lon)
        if lat_raw is None or lon_raw is None:
            continue
        lat = (
            _from_descriptor(lat_raw)
            if isinstance(lat_raw, dict)
            else _to_float_seq(lat_raw)
        )
        lon = (
            _from_descriptor(lon_raw)
            if isinstance(lon_raw, dict)
            else _to_float_seq(lon_raw, ("lon", "longitude", "x", "value"))
        )
        if lat and lon:
            return lat, lon

    for path in (
        "meta.grid.points",
        "meta.points",
        "om.grid.points",
        "om.points",
        "coords.points",
        "grid.points",
    ):
        seq = dig(j, path)
        if isinstance(seq, (list, tuple)) and seq:
            lat, lon = _from_points(seq)
            if lat and lon:
                return lat, lon

    for p_lat, p_lon in array_pairs:
        lat_raw, lon_raw = dig(j, p_lat), dig(j, p_lon)
        if isinstance(lat_raw, dict) and isinstance(lon_raw, dict):
            lat, lon = _from_descriptor(lat_raw), _from_descriptor(lon_raw)
            if lat and lon:
                return lat, lon

    for node in (j.get("meta", {}) or {}, j):
        lat, lon = _from_bbox(node)
        if lat and lon:
            return lat, lon

    grid = j.get("grid")
    if isinstance(grid, dict):
        lat_raw, lon_raw = grid.get("lat"), grid.get("lon")
        lat = (
            _from_descriptor(lat_raw)
            if isinstance(lat_raw, dict)
            else _to_float_seq(lat_raw)
        )
        lon = (
            _from_descriptor(lon_raw)
            if isinstance(lon_raw, dict)
            else _to_float_seq(lon_raw, ("lon", "longitude", "x", "value"))
        )
        if lat and lon:
            return lat, lon
        pts = grid.get("points")
        if isinstance(pts, (list, tuple)):
            lat, lon = _from_points(pts)
            if lat and lon:
                return lat, lon

    return None, None


def _extract_grid_points(j):
    """
    Returns list[(lat, lon)] in the *exact* order baselines are flattened.

    Priority:
      1) meta.paired_lat / meta.paired_lon  -> zipped pairs (batch subsets)
      2) Explicit arrays (via _get_lat_lon_lists) -> Cartesian product (flatten_grid)
      3) meta.lat_axis / meta.lon_axis -> Cartesian product (full axes)
    """
    meta = j.get("meta", {}) or {}
    p_lat = meta.get("paired_lat")
    p_lon = meta.get("paired_lon")
    if (
        isinstance(p_lat, list)
        and isinstance(p_lon, list)
        and len(p_lat) == len(p_lon)
        and len(p_lat) > 0
    ):
        try:
            return [(float(la), float(lo)) for la, lo in zip(p_lat, p_lon)]
        except Exception:
            pass

    lat, lon = _get_lat_lon_lists(j)
    if lat and lon:
        return flatten_grid(lat, lon)

    lat_axis = meta.get("lat_axis")
    lon_axis = meta.get("lon_axis")
    if (
        isinstance(lat_axis, list)
        and isinstance(lon_axis, list)
        and lat_axis
        and lon_axis
    ):
        try:
            lat = [float(x) for x in lat_axis]
            lon = [float(x) for x in lon_axis]
            return flatten_grid(lat, lon)
        except Exception:
            pass
    return None


# -------------------------
# New: robust baseline parsing
# -------------------------
def _parse_times_from_hourly_block(block):
    """block = {'time': [...], var1:[...], ...}"""
    if not isinstance(block, dict) or "time" not in block:
        return None
    try:
        return [dt.datetime.fromisoformat(t.replace("Z", "")) for t in block["time"]]
    except Exception:
        return None


def _extract_times(j):
    """Try om (dict/list) first; fallback to meta.start/end."""
    om = j.get("om")
    # om as dict with 'hourly'
    if isinstance(om, dict) and "hourly" in om:
        t = _parse_times_from_hourly_block(om["hourly"])
        if t:
            return t
    # om as list of per-point dicts
    if isinstance(om, list) and om:
        hb = om[0].get("hourly", {})
        t = _parse_times_from_hourly_block(hb)
        if t:
            return t
    # fallback meta.start/end (exclusive end)
    m = j.get("meta", {}) or {}
    try:
        t0 = dt.datetime.fromisoformat(m["start"])
        t1 = dt.datetime.fromisoformat(m["end"])
        return [
            t0 + dt.timedelta(hours=h)
            for h in range(int((t1 - t0).total_seconds() // 3600))
        ]
    except Exception:
        return None


def _build_matrix_from_points(point_list, name, times, grid):
    """
    point_list: list of {latitude, longitude, hourly:{ name: [T], ... }}
    Returns arr shape (T, G) matching grid order.
    """
    T, G = len(times), len(grid)

    # map (rounded lat,lon) -> vector
    def _r(x):
        return round(float(x), 6)

    series_map = {}
    for p in point_list:
        la = p.get("latitude")
        lo = p.get("longitude")
        hb = p.get("hourly", {}) or {}
        vals = hb.get(name)
        if la is None or lo is None or vals is None:
            continue
        v = np.asarray(vals, dtype=float)
        if v.size != T:
            # if time lengths vary, try to align via reported time when present
            p_times = _parse_times_from_hourly_block(hb)
            if p_times and len(p_times) == v.size:
                # simple nearest-hour alignment to global 'times'
                aligned = np.empty(T, dtype=float)
                for i, tt in enumerate(times):
                    j = int(
                        np.argmin([abs((tt - pt).total_seconds()) for pt in p_times])
                    )
                    aligned[i] = float(v[j])
                v = aligned
            else:
                raise RuntimeError(
                    f"point ({la},{lo}) var {name} length {v.size} != T={T}"
                )
        series_map[(_r(la), _r(lo))] = v

    out = np.empty((T, G), dtype=float)
    missing = 0
    for gi, (la, lo) in enumerate(grid):
        key = (_r(la), _r(lo))
        if key not in series_map:
            missing += 1
            # If missing, fill NaN and let downstream skip/handle
            out[:, gi] = np.nan
        else:
            out[:, gi] = series_map[key]
    if missing and missing == G:
        raise KeyError(f"no points matched grid for var {name}")
    return out


def _parse_baseline(j, key, times, grid, required_vars):
    """
    key: 'om' or 'ifs'
    Returns dict: var_name -> (T,G) ndarray or None if not available.
    Accepts dict-with-hourly or list-of-points schema.
    """
    blk = j.get(key, None)
    if blk is None:
        return {v: None for v in required_vars}

    T, G = len(times), len(grid)

    if isinstance(blk, dict) and "hourly" in blk:
        hourly = blk["hourly"] or {}
        out = {}
        for v in required_vars:
            arr = hourly.get(v)
            if arr is None:
                out[v] = None
            else:
                out[v] = as_tg(arr, T, G if np.ndim(arr) > 1 else 1)  # tolerate 1D
                if out[v].shape == (T, 1) and G > 1:
                    # If it's truly a single point but grid > 1, broadcast if that's expected
                    out[v] = np.repeat(out[v], G, axis=1)
        return out

    if isinstance(blk, list):
        out = {}
        for v in required_vars:
            try:
                out[v] = _build_matrix_from_points(blk, v, times, grid)
            except KeyError:
                out[v] = None
        return out

    # Unknown schema
    return {v: None for v in required_vars}


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

    needed = [
        "temperature_2m",
        "dew_point_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_100m",
        "wind_direction_100m",
    ]

    for p in files:
        try:
            with open(p, "r") as f:
                j = json.load(f)
        except Exception as e:
            print(f"[pair] skip {os.path.basename(p)} (invalid JSON: {e})")
            continue

        # Grid (ordered)
        grid = _extract_grid_points(j)
        if not grid:
            print(f"[pair] skip {os.path.basename(p)} (unable to parse lat/lon grid)")
            continue

        # Times
        times = _extract_times(j)
        if not times:
            print(f"[pair] skip {os.path.basename(p)} (unable to parse times)")
            continue
        T, G = len(times), len(grid)
        if T == 0 or G == 0:
            print(f"[pair] skip {os.path.basename(p)} (empty times or grid)")
            continue

        t0 = times[0]
        leads = np.array(
            [(t - t0).total_seconds() / 3600.0 for t in times], dtype=float
        )
        hods = np.array([t.hour + t.minute / 60 for t in times], dtype=float)

        # Parse baselines (robust to both schemas)
        om_vars = _parse_baseline(j, "om", times, grid, needed)
        ifs_vars = _parse_baseline(j, "ifs", times, grid, needed)

        # Ensure required OM variables present
        missing_keys = [k for k in needed if om_vars.get(k) is None]
        if missing_keys:
            print(
                f"[pair] skip {os.path.basename(p)} (missing OM keys: {missing_keys})"
            )
            continue

        # Aliases
        b_t2m_om = om_vars["temperature_2m"]
        b_td2m_om = om_vars["dew_point_2m"]
        b_psfc_om = om_vars["surface_pressure"]
        b_tp_om = om_vars["precipitation"]
        b_wspd_om = om_vars["wind_speed_100m"]
        b_wdir_om = om_vars["wind_direction_100m"]

        b_t2m_ifs = ifs_vars.get("temperature_2m")
        b_td2m_ifs = ifs_vars.get("dew_point_2m")
        b_psfc_ifs = ifs_vars.get("surface_pressure")
        b_tp_ifs = ifs_vars.get("precipitation")
        b_wspd_ifs = ifs_vars.get("wind_speed_100m")
        b_wdir_ifs = ifs_vars.get("wind_direction_100m")

        # Broadcast single-column OM to G if needed (defensive)
        def _ensure_TG(a):
            a = np.asarray(a)
            if a.shape == (T,):
                a = a[:, None]
            if a.shape == (T, 1) and G > 1:
                a = np.repeat(a, G, axis=1)
            if a.shape != (T, G):
                raise RuntimeError(f"baseline shape mismatch {a.shape}, want {(T,G)}")
            return a

        try:
            b_t2m_om = _ensure_TG(b_t2m_om)
            b_td2m_om = _ensure_TG(b_td2m_om)
            b_psfc_om = _ensure_TG(b_psfc_om)
            b_tp_om = _ensure_TG(b_tp_om)
            b_wspd_om = _ensure_TG(b_wspd_om)
            b_wdir_om = _ensure_TG(b_wdir_om)
            if b_t2m_ifs is not None:
                b_t2m_ifs = _ensure_TG(b_t2m_ifs)
            if b_td2m_ifs is not None:
                b_td2m_ifs = _ensure_TG(b_td2m_ifs)
            if b_psfc_ifs is not None:
                b_psfc_ifs = _ensure_TG(b_psfc_ifs)
            if b_tp_ifs is not None:
                b_tp_ifs = _ensure_TG(b_tp_ifs)
            if b_wspd_ifs is not None:
                b_wspd_ifs = _ensure_TG(b_wspd_ifs)
            if b_wdir_ifs is not None:
                b_wdir_ifs = _ensure_TG(b_wdir_ifs)
        except Exception as e:
            print(f"[pair] skip {os.path.basename(p)} (shape issue: {e})")
            continue

        # --- robust coord/time helpers ------------------------------------------------
        def _coord_values(ds, candidates):
            """Return the first matching coordinate's numpy array for any of the candidate names."""
            for name in candidates:
                if name in ds.coords:
                    return ds.coords[name].values
                if (
                    name in ds.dims and name in ds
                ):  # sometimes coords also appear as data vars
                    try:
                        return ds[name].values
                    except Exception:
                        pass
            raise KeyError(
                f"None of coord names {candidates} found in dataset: {list(ds.coords)}"
            )

        def _time_list(ds):
            """
            Return a Python list[datetime] for the dataset's time-like coordinate.
            Tries common names and tolerates non-pandas indexes.
            """
            # prefer coords; fall back to dims-as-vars
            tvals = _coord_values(ds, ("time", "valid_time", "t"))
            # normalize to python datetimes
            try:
                import pandas as _pd

                return [
                    ts.to_pydatetime().replace(tzinfo=None)
                    for ts in _pd.to_datetime(tvals)
                ]
            except Exception:
                # final fallback via numpy
                return [
                    np.datetime64(x)
                    .astype("datetime64[ms]")
                    .astype(object)
                    .replace(tzinfo=None)
                    for x in tvals
                ]

        def _lat_vals(ds):
            return _coord_values(ds, ("latitude", "lat", "y"))

        def _lon_vals(ds):
            return _coord_values(ds, ("longitude", "lon", "x"))

        def _wrap_lon_for_coords(lon_coords, target_lon):
            """
            If dataset longitudes are 0..360 and target is -180..180, wrap target into same range.
            """
            lon_coords = np.asarray(lon_coords)
            lo = float(target_lon)
            if lon_coords.min() >= 0.0 and lon_coords.max() > 180.0:
                # dataset uses 0..360
                if lo < 0.0:
                    lo = (lo + 360.0) % 360.0
            else:
                # dataset likely -180..180; also map 0..360 targets back if needed
                if lo > 180.0:
                    lo = ((lo + 180.0) % 360.0) - 180.0
            return lo

        # --- replacement: robust ERA5 nearest sampler --------------------------------
        def nearest_truth(ds, var, grid, times):
            lats = _lat_vals(ds)
            lons = _lon_vals(ds)
            tlist = _time_list(ds)  # python datetimes

            T, G = len(times), len(grid)
            out = np.empty((T, G), dtype=float)

            # precompute for speed
            lats = np.asarray(lats)
            lons = np.asarray(lons)

            for gi, (la, lo) in enumerate(grid):
                # wrap longitude if grid uses -180..180 but dataset is 0..360 (or vice versa)
                lo_adj = _wrap_lon_for_coords(lons, lo)
                ilat = int(np.argmin(np.abs(lats - la)))
                ilon = int(np.argmin(np.abs(lons - lo_adj)))
                for ti, tt in enumerate(times):
                    it = int(
                        np.argmin([abs((tt - t0).total_seconds()) for t0 in tlist])
                    )
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
                    bifsv = (
                        float(bifs[ti, gi])
                        if (bifs is not None and np.isfinite(bifs[ti, gi]))
                        else np.nan
                    )
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
                            "time": times[ti].isoformat(),
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
