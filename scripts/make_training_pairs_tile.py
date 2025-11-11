#!/usr/bin/env python
import os, json, glob, argparse, datetime as dt, logging, warnings
import numpy as np, pandas as pd, xarray as xr

OM_DIR = os.path.join("data", "om_baseline")
OUT_DIR = os.path.join("data", "train_tiles")
os.makedirs(OUT_DIR, exist_ok=True)

log = logging.getLogger("make_pairs")


# -------------------------
# Utilities
# -------------------------
def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # xarray / numexpr can be noisy
    warnings.filterwarnings("ignore", message=".*lazy array.*")
    warnings.filterwarnings("ignore", message="Mean of empty slice.*")


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


# -------------------------
# Manifest (processed OM files per tile)
# -------------------------
def _manifest_path(tile_id: str) -> str:
    return os.path.join(OUT_DIR, f"{tile_id}__processed_om.txt")


def _load_processed(tile_id: str) -> set[str]:
    path = _manifest_path(tile_id)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception:
        return set()


def _append_processed(tile_id: str, basenames: list[str]) -> None:
    if not basenames:
        return
    path = _manifest_path(tile_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for b in basenames:
            f.write(b + "\n")


# -------------------------
# Robust lat/lon parsing
# -------------------------
def _get_lat_lon_lists(j):
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
                    assert n_v > 0
                    return list(np.linspace(a_v, b_v, n_v))
                except Exception:
                    pass
        if all(k in desc for k in ("start", "step", "count")):
            try:
                start = _as_float(desc["start"])
                step = _as_float(desc["step"])
                cnt = int(desc["count"])
                assert cnt > 0
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
                out.append(float(x))
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
                            got = float(x[kk])
                            break
                        except Exception:
                            pass
                if got is not None:
                    out.append(got)
                    continue
            if isinstance(x, (list, tuple)) and len(x) >= 1:
                try:
                    out.append(float(x[0]))
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
    meta = j.get("meta", {}) or {}
    p_lat, p_lon = meta.get("paired_lat"), meta.get("paired_lon")
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
    lat_axis, lon_axis = meta.get("lat_axis"), meta.get("lon_axis")
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
# Baseline parsing (snap requested grid to nearest available OM/IFS points)
# -------------------------
def _parse_times_from_hourly_block(block):
    if not isinstance(block, dict) or "time" not in block:
        return None
    try:
        return [dt.datetime.fromisoformat(t.replace("Z", "")) for t in block["time"]]
    except Exception:
        return None


def _extract_times(j):
    # prefer meta start/end (request-based), fall back to first OM item
    m = j.get("meta", {}) or {}
    try:
        t0 = dt.datetime.fromisoformat(m["start"])
        t1 = dt.datetime.fromisoformat(m["end"])
        return [
            t0 + dt.timedelta(hours=h)
            for h in range(int((t1 - t0).total_seconds() // 3600))
        ]
    except Exception:
        pass
    om = j.get("om")
    if isinstance(om, dict) and "hourly" in om:
        t = _parse_times_from_hourly_block(om["hourly"])
        if t:
            return t
    if isinstance(om, list) and om:
        hb = om[0].get("hourly", {})
        t = _parse_times_from_hourly_block(hb)
        if t:
            return t
    return None


def _lon_canon(l):
    # Canonicalize to [-180, 180), mapping +180 -> -180
    x = ((float(l) + 180.0) % 360.0) - 180.0
    return -180.0 if abs(x - 180.0) < 1e-9 else x


def _lon_diff_deg(a, b):
    """Shortest signed lon difference in degrees with ±180 equivalence."""
    A = _lon_canon(a)
    B = _lon_canon(b)
    d = ((A - B + 540.0) % 360.0) - 180.0
    # Make 180 act like -180 to avoid edge mismatches
    return -180.0 if abs(d - 180.0) < 1e-9 else d


def _nearest_idx_points(points_lat, points_lon, req_lat, req_lon):
    """Return index of nearest point by great-circle-aware degree metric."""
    # weight longitude by cos(lat) to approximate distance in degrees
    lat_arr = np.asarray(points_lat, dtype=float)
    lon_arr = np.asarray(points_lon, dtype=float)

    dlat = lat_arr - float(req_lat)
    dlon = np.array([_lon_diff_deg(l, req_lon) for l in lon_arr], dtype=float)
    w = np.cos(np.deg2rad(float(req_lat)))
    dist2 = dlat * dlat + (w * dlon) * (w * dlon)
    return int(np.argmin(dist2)), float(np.min(dist2))


def _build_matrix_from_points(point_list, name, times, grid, tol_deg=0.6):
    """
    Build (T,G) array on requested grid by snapping each requested (lat,lon)
    to the nearest available returned point (lat',lon') within tol_deg.
    Adds a polar rule: when |lat| >= 89.5°, ignore longitude in distance.
    """
    T, G = len(times), len(grid)

    # Collect available coordinates and series
    pts_lat, pts_lon, series = [], [], []
    for p in point_list:
        la = p.get("latitude")
        lo = p.get("longitude")
        hb = p.get("hourly", {}) or {}
        vals = hb.get(name)
        if la is None or lo is None or vals is None:
            continue

        v = np.asarray(vals, dtype=float)
        p_times = _parse_times_from_hourly_block(hb)

        # Align time length to requested axis if needed
        if p_times and len(p_times) == v.size and v.size != T:
            aligned = np.empty(T, dtype=float)
            for i, tt in enumerate(times):
                j = int(np.argmin([abs((tt - pt).total_seconds()) for pt in p_times]))
                aligned[i] = float(v[j])
            v = aligned
        elif v.size != T:
            # Can't align → skip this point
            continue

        pts_lat.append(float(la))
        pts_lon.append(_lon_canon(lo))
        series.append(v.astype(float))

    if not series:
        raise KeyError(f"no points carried var {name}")

    pts_lat = np.asarray(pts_lat)
    pts_lon = np.asarray(pts_lon)
    out = np.full((T, G), np.nan, dtype=float)
    tol2 = tol_deg**2

    for gi, (la_req, lo_req) in enumerate(grid):
        la_req = float(la_req)
        lo_req = _lon_canon(lo_req)

        # Polar snap: ignore longitude when near the poles
        if abs(la_req) >= 89.5:
            dlat = np.abs(pts_lat - la_req)
            idx = int(np.argmin(dlat))
            d2 = float(dlat[idx] ** 2)  # only latitude contributes
        else:
            dlat = pts_lat - la_req
            dlon = np.array([_lon_diff_deg(L, lo_req) for L in pts_lon], dtype=float)
            w = np.cos(
                np.deg2rad(la_req)
            )  # shrink longitudinal distance away from equator
            dist2 = dlat * dlat + (w * dlon) * (w * dlon)
            idx = int(np.argmin(dist2))
            d2 = float(dist2[idx])

        if d2 <= tol2:
            out[:, gi] = series[idx]
        # else: leave NaN; caller will handle (and we’ll log in the shape check)
    # If absolutely nothing matched, signal failure
    if np.all(~np.isfinite(out)):
        raise KeyError(f"no points matched requested grid for var {name}")
    return out


def _parse_baseline(j, key, times, grid, required_vars):
    blk = j.get(key, None)
    if blk is None:
        return {v: None for v in required_vars}
    T, G = len(times), len(grid)

    # Case A: block is a dict with "hourly" already (rare in your dumps)
    if isinstance(blk, dict) and "hourly" in blk:
        hourly = blk["hourly"] or {}
        out = {}
        for v in required_vars:
            arr = hourly.get(v)
            out[v] = (
                None if arr is None else as_tg(arr, T, G if np.ndim(arr) > 1 else 1)
            )
            if out[v] is not None and out[v].shape == (T, 1) and G > 1:
                out[v] = np.repeat(out[v], G, axis=1)
        return out

    # Case B: list of point objects (typical OM/IFS baseline you saved)
    if isinstance(blk, list):
        out = {}
        for v in required_vars:
            try:
                out[v] = _build_matrix_from_points(blk, v, times, grid)
            except KeyError:
                out[v] = None
        return out

    return {v: None for v in required_vars}


# -------------------------
# ERA5 variable resolution + sampling
# -------------------------
VAR_ALIASES = {
    "t2m": ["t2m", "2m_temperature", "temperature_2m"],
    "d2m": [
        "d2m",
        "2m_dewpoint_temperature",
        "dew_point_2m",
        "dewpoint_temperature_2m",
    ],
    "tp": ["tp", "total_precipitation", "precipitation"],
    "sp": ["sp", "surface_pressure"],
    "u100": ["u100", "100m_u_component_of_wind"],
    "v100": ["v100", "100m_v_component_of_wind"],
    "u10": ["u10", "10m_u_component_of_wind"],
    "v10": ["v10", "10m_v_component_of_wind"],
}


def _find_var_name(ds: xr.Dataset, key: str) -> str:
    aliases = VAR_ALIASES.get(key, [key])
    data_vars_lower = {name.lower(): name for name in ds.data_vars}
    for cand in aliases:
        name = data_vars_lower.get(cand.lower())
        if name is not None:
            log.debug(
                f"[resolve] {key} -> {name} (data_var in {getattr(ds, 'encoding', {}).get('source','<mem>')})"
            )
            return name
    coords_lower = {name.lower(): name for name in ds.coords}
    for cand in aliases:
        name = coords_lower.get(cand.lower())
        if name is not None:
            log.debug(f"[resolve] {key} -> {name} (coord)")
            return name
    raise KeyError(
        f"No variable named {aliases!r}. Variables include {list(ds.data_vars)}"
    )


def _maybe_squeeze_members(da: xr.DataArray) -> xr.DataArray:
    for extra in ("expver", "number"):
        if extra in da.dims:
            log.debug(f"[squeeze] selecting last index for dim '{extra}'")
            da = da.isel({extra: -1})
    return da.squeeze(drop=True)


def _coord_values(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return ds.coords[name].values
        if name in ds.dims and name in ds:
            try:
                return ds[name].values
            except Exception:
                pass
    raise KeyError(f"None of coord names {candidates} found; coords: {list(ds.coords)}")


def _time_list(ds):
    tvals = _coord_values(ds, ("time", "valid_time", "t"))
    try:
        import pandas as _pd

        return [
            ts.to_pydatetime().replace(tzinfo=None) for ts in _pd.to_datetime(tvals)
        ]
    except Exception:
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
    lon_coords = np.asarray(lon_coords)
    lo = float(target_lon)
    if lon_coords.min() >= 0.0 and lon_coords.max() > 180.0:
        if lo < 0.0:
            lo = (lo + 360.0) % 360.0
    else:
        if lo > 180.0:
            lo = ((lo + 180.0) % 360.0) - 180.0
    return lo


def _has_var(ds, key):
    try:
        _find_var_name(ds, key)
        return True
    except KeyError:
        return False


def _prefer_dataset_for(era5_land, era5_single, key):
    if _has_var(era5_land, key):
        log.debug(f"[choose-ds] {key}: ERA5-Land")
        return era5_land
    if _has_var(era5_single, key):
        log.debug(f"[choose-ds] {key}: ERA5 Single-levels")
        return era5_single
    log.debug(f"[choose-ds] {key}: not found; default ERA5 Single-levels")
    return era5_single


def nearest_truth(ds, key, grid, times):
    vname = _find_var_name(ds, key)
    da = _maybe_squeeze_members(ds[vname])

    lats = _lat_vals(ds)
    lons = _lon_vals(ds)
    tlist = _time_list(ds)
    T, G = len(times), len(grid)
    out = np.empty((T, G), dtype=float)
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    # Precompute time index map to requested times
    t_idx = []
    for tt in times:
        t_idx.append(int(np.argmin([abs((tt - t0).total_seconds()) for t0 in tlist])))

    for gi, (la, lo) in enumerate(grid):
        lo_adj = _wrap_lon_for_coords(lons, lo)
        ilat = int(np.argmin(np.abs(lats - la)))
        ilon = int(np.argmin(np.abs(lons - lo_adj)))
        out[:, gi] = da.isel(
            latitude=ilat, longitude=ilon, time=xr.DataArray(t_idx)
        ).values
    return out


# -------------------------
# Robust parquet writer (with append+dedupe)
# -------------------------
def safe_to_parquet(
    df: pd.DataFrame, path: str, append: bool = False, subset=("lat", "lon", "time")
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _write(df_final: pd.DataFrame):
        engines = []
        try:
            import pyarrow  # noqa: F401

            engines.append(("pyarrow", {}))
        except Exception:
            pass
        try:
            import fastparquet  # noqa: F401

            engines.append(("fastparquet", {}))
        except Exception:
            pass

        last_err = None
        for eng, kwargs in engines:
            try:
                log.debug(f"[parquet] writing {path} using engine={eng}")
                df_final.to_parquet(path, engine=eng, index=False, **kwargs)
                return
            except Exception as e:
                log.warning(f"[parquet] engine={eng} failed: {e!r}")
                last_err = e

        # Final fallback: CSV
        csv_path = os.path.splitext(path)[0] + ".csv"
        df_final.to_csv(csv_path, index=False)
        log.error(
            f"[parquet] all parquet engines failed; wrote CSV fallback: {csv_path}"
        )
        if last_err:
            log.debug(f"[parquet] last error: {last_err!r}")

    if append and os.path.exists(path):
        try:
            prev = pd.read_parquet(path)
            combined = pd.concat([prev, df], ignore_index=True)
            if subset:
                combined = combined.drop_duplicates(subset=list(subset), keep="last")
            _write(combined)
            return
        except Exception as e:
            log.warning(
                f"[parquet] append failed to read existing {path}: {e!r}; rewriting fresh"
            )
    _write(df)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-single", required=True)
    ap.add_argument("--era5-land", required=True)
    ap.add_argument("--tile-id", required=True)
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore manifest and rebuild from all OM JSONs",
    )
    args = ap.parse_args()
    setup_logging(args.log_level)

    log.info(f"Loading ERA5 Single: {args.era5_single}")
    era5_single = xr.open_dataset(args.era5_single)
    log.info(f"Loading ERA5-Land:  {args.era5_land}")
    era5_land = xr.open_dataset(args.era5_land)

    rows = {k: [] for k in ["t2m", "td2m", "psfc", "tp", "wspd100", "wdir100"]}

    files = sorted(glob.glob(os.path.join(OM_DIR, "omifs_*.json")))
    if not files:
        raise SystemExit("No OM baseline files found")

    processed = set() if args.force else _load_processed(args.tile_id)
    new_processed: list[str] = []

    needed = [
        "temperature_2m",
        "dew_point_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_100m",
        "wind_direction_100m",
    ]

    for p in files:
        base = os.path.basename(p)
        if base in processed:
            log.info(f"[pair] skip already processed: {base}")
            continue

        try:
            with open(p, "r") as f:
                j = json.load(f)
        except Exception as e:
            log.warning(f"[pair] skip {base} (invalid JSON: {e})")
            continue

        grid = _extract_grid_points(j)
        if not grid:
            log.warning(f"[pair] skip {base} (unable to parse lat/lon grid)")
            continue

        times = _extract_times(j)
        if not times:
            log.warning(f"[pair] skip {base} (unable to parse times)")
            continue

        T, G = len(times), len(grid)
        if T == 0 or G == 0:
            log.warning(f"[pair] skip {base} (empty times or grid)")
            continue

        log.info(f"[pair] {base}: T={T}, G={G}")

        t0 = times[0]
        leads = np.array(
            [(t - t0).total_seconds() / 3600.0 for t in times], dtype=float
        )
        hods = np.array([t.hour + t.minute / 60 for t in times], dtype=float)

        om_vars = _parse_baseline(j, "om", times, grid, needed)
        ifs_vars = _parse_baseline(j, "ifs", times, grid, needed)

        missing_keys = [k for k in needed if om_vars.get(k) is None]
        if missing_keys:
            log.warning(f"[pair] skip {base} (missing OM keys: {missing_keys})")
            continue

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

        def _ensure_TG(a):
            a = np.asarray(a)
            if a.ndim == 1:
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
            log.warning(f"[pair] skip {base} (shape issue: {e})")
            continue

        # ERA5 truth sampling on requested grid/time
        try:
            ds_t2m = _prefer_dataset_for(era5_land, era5_single, "t2m")
            ds_d2m = _prefer_dataset_for(era5_land, era5_single, "d2m")
            ds_tp = _prefer_dataset_for(era5_land, era5_single, "tp")
            ds_sp = _prefer_dataset_for(era5_land, era5_single, "sp")
            ds_u100 = _prefer_dataset_for(era5_land, era5_single, "u100")
            ds_v100 = _prefer_dataset_for(era5_land, era5_single, "v100")

            t2m_truth = to_celsius(nearest_truth(ds_t2m, "t2m", grid, times))
            td2m_truth = to_celsius(nearest_truth(ds_d2m, "d2m", grid, times))
            tp_truth = m_to_mm(nearest_truth(ds_tp, "tp", grid, times))
            sp_truth = pa_to_hpa(nearest_truth(ds_sp, "sp", grid, times))
            u100 = nearest_truth(ds_u100, "u100", grid, times)
            v100 = nearest_truth(ds_v100, "v100", grid, times)
            wspd_truth = ms_to_kmh(np.hypot(u100, v100))
            wdir_truth = (np.degrees(np.arctan2(-u100, -v100)) + 360.0) % 360.0
        except Exception as e:
            log.warning(f"[pair] skip {base} (ERA5 sample error: {e})")
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

        log.info(f"paired {base} → {args.tile_id}")
        new_processed.append(base)

    # Write per-tile outputs (append + dedupe by lat,lon,time)
    uniq_key = ("lat", "lon", "time")
    for key, outname in [
        ("t2m", "t2m"),
        ("td2m", "td2m"),
        ("psfc", "psfc"),
        ("tp", "tp"),
        ("wspd100", "wspd100"),
        ("wdir100", "wdir100"),
    ]:
        df = pd.DataFrame(rows[key])
        out_path = os.path.join(OUT_DIR, f"{args.tile_id}__{outname}.parquet")
        if not df.empty:
            safe_to_parquet(df, out_path, append=True, subset=uniq_key)
            log.info(f"saved {args.tile_id} {outname} -> {out_path} (append+dedupe)")
        else:
            log.info(f"[pair] no new rows for {outname} on {args.tile_id}")

    _append_processed(args.tile_id, new_processed)
    if new_processed:
        log.info(
            f"[manifest] recorded {len(new_processed)} new OM files for {args.tile_id}"
        )
    else:
        log.info(f"[manifest] no new OM files processed for {args.tile_id}")


if __name__ == "__main__":
    main()
