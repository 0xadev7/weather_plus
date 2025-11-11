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

    Accepts all of these (and mixes of them):
      A) Explicit arrays:
         - meta.lat / meta.lon
         - meta.lats / meta.lons
         - meta.latitude / meta.longitude
         - meta.grid.lat / meta.grid.lon
         - coords.lat / coords.lon
         - om.latitude / om.longitude
         - om.meta.lat / om.meta.lon
         - om.grid.lat / om.grid.lon
         - grid.lat / grid.lon
      B) Points:
         - *.grid.points = [{lat:.., lon:..}, ...]
         - *.points = [...]
      C) Descriptors (build arrays from a spec):
         - {values:[...]}
         - {start:.., stop:.., num:..} or {min:.., max:.., n:..}
         - {start:.., step:.., count:..} (inclusive of endpoints)
      D) Bounding box + resolution:
         - meta.bbox + meta.nx/meta.ny OR meta.n
         - meta.bbox with keys: north/south/east/west OR lat_min/lat_max/lon_min/lon_max
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

    # ---- descriptor -> sequence ------------------------------------------------
    def _from_descriptor(desc, axis="lat"):
        """
        Turn a dict descriptor into a float list.
        Supported:
          - {'values': [...]}
          - {'start': a, 'stop': b, 'num': N}
          - {'min': a, 'max': b, 'n': N}
          - {'start': a, 'step': s, 'count': N}
        """
        if not isinstance(desc, dict):
            return None
        # direct values
        if "values" in desc and isinstance(desc["values"], (list, tuple)):
            out = []
            for v in desc["values"]:
                try:
                    out.append(_as_float(v))
                except Exception:
                    continue
            return out if out else None

        # start/stop/num | min/max/n
        keys1 = [("start", "stop", "num"), ("min", "max", "n")]
        for a, b, n in keys1:
            if a in desc and b in desc and n in desc:
                try:
                    a_v, b_v = _as_float(desc[a]), _as_float(desc[b])
                    n_v = int(desc[n])
                    if n_v <= 0:
                        return None
                    return list(np.linspace(a_v, b_v, n_v))
                except Exception:
                    pass

        # start/step/count
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

        # sometimes nested: {'array': [...]} or {'data': [...]}
        for k in ("array", "data"):
            if k in desc and isinstance(desc[k], (list, tuple)):
                try:
                    return [_as_float(v) for v in desc[k]]
                except Exception:
                    pass

        return None

    # ---- bbox + resolution -----------------------------------------------------
    def _from_bbox(meta):
        """
        Build lat/lon lists from a bbox + nx/ny (or 'n' square).
        bbox may be:
          - dict with north/south/east/west
          - dict with lat_min/lat_max/lon_min/lon_max
          - list/tuple [lat_min, lat_max, lon_min, lon_max] (Open-Meteo-ish)
        Resolution:
          - nx/ny or n
        """
        if not isinstance(meta, dict):
            return None, None

        bbox = meta.get("bbox")
        if bbox is None:
            return None, None

        def _parse_bbox(bb):
            if isinstance(bb, dict):
                # named keys
                if all(k in bb for k in ("south", "north", "west", "east")):
                    return (
                        _as_float(bb["south"]),
                        _as_float(bb["north"]),
                        _as_float(bb["west"]),
                        _as_float(bb["east"]),
                    )
                if all(k in bb for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
                    return (
                        _as_float(bb["lat_min"]),
                        _as_float(bb["lat_max"]),
                        _as_float(bb["lon_min"]),
                        _as_float(bb["lon_max"]),
                    )
            elif isinstance(bb, (list, tuple)) and len(bb) == 4:
                # assume [lat_min, lat_max, lon_min, lon_max]
                return (
                    _as_float(bb[0]),
                    _as_float(bb[1]),
                    _as_float(bb[2]),
                    _as_float(bb[3]),
                )
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
                # also allow 'res' or 'resolution' as approximate count per axis
                res = meta.get("res") or meta.get("resolution")
                if res is not None:
                    nx = ny = int(res)
            nx = int(nx) if nx is not None else None
            ny = int(ny) if ny is not None else None
        except Exception:
            nx = ny = None

        if not nx or not ny:
            return None, None

        # Build inclusive ranges
        lat_list = list(np.linspace(lat_min, lat_max, ny))
        lon_list = list(np.linspace(lon_min, lon_max, nx))
        return lat_list, lon_list

    # ---- heterogeneous list -> floats -----------------------------------------
    def _to_float_seq(seq, key_candidates=("lat", "latitude", "y", "value")):
        out = []
        for x in seq:
            try:
                out.append(_as_float(x))
                continue
            except Exception:
                pass
            # descriptors embedded in the list
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
                            la = _as_float(p[k])
                            break
                        except Exception:
                            pass
                for k in ("lon", "longitude", "x"):
                    if k in p:
                        try:
                            lo = _as_float(p[k])
                            break
                        except Exception:
                            pass
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    la = _as_float(p[0])
                    lo = _as_float(p[1])
                except Exception:
                    pass
            if la is not None and lo is not None:
                lats.append(la)
                lons.append(lo)
        if lats and lons:
            return _ensure_sorted_unique(lats), _ensure_sorted_unique(lons)
        return None, None

    # 1) Try explicit array pairs first (and allow descriptors at those paths)
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

        # if dict descriptors
        if isinstance(lat_raw, dict):
            lat = _from_descriptor(lat_raw, axis="lat")
        else:
            lat = _to_float_seq(lat_raw)

        if isinstance(lon_raw, dict):
            lon = _from_descriptor(lon_raw, axis="lon")
        else:
            lon = _to_float_seq(
                lon_raw, key_candidates=("lon", "longitude", "x", "value")
            )

        if lat and lon:
            return lat, lon

    # 2) Points-style schemas
    point_candidates = [
        "meta.grid.points",
        "meta.points",
        "om.grid.points",
        "om.points",
        "coords.points",
        "grid.points",
    ]
    for path in point_candidates:
        seq = dig(j, path)
        if isinstance(seq, (list, tuple)) and seq:
            lat, lon = _from_points(seq)
            if lat and lon:
                return lat, lon

    # 3) Descriptor-only lat/lon at common locations (e.g., meta.lat is a descriptor)
    for p_lat, p_lon in array_pairs:
        lat_raw, lon_raw = dig(j, p_lat), dig(j, p_lon)
        if isinstance(lat_raw, dict) and isinstance(lon_raw, dict):
            lat = _from_descriptor(lat_raw, axis="lat")
            lon = _from_descriptor(lon_raw, axis="lon")
            if lat and lon:
                return lat, lon

    # 4) BBox + resolution on meta or top-level
    for node in (j.get("meta", {}), j):
        lat, lon = _from_bbox(node)
        if lat and lon:
            return lat, lon

    # 5) Some dumps put everything under a single 'grid' dict with descriptors
    grid = j.get("grid")
    if isinstance(grid, dict):
        lat_raw, lon_raw = grid.get("lat"), grid.get("lon")
        lat = (
            _from_descriptor(lat_raw, axis="lat")
            if isinstance(lat_raw, dict)
            else _to_float_seq(lat_raw)
        )
        lon = (
            _from_descriptor(lon_raw, axis="lon")
            if isinstance(lon_raw, dict)
            else _to_float_seq(
                lon_raw, key_candidates=("lon", "longitude", "x", "value")
            )
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
