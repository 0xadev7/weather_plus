#!/usr/bin/env python
import os, json, glob, argparse, datetime as dt, logging, warnings
import fsspec
import numpy as np, pandas as pd, xarray as xr
import zipfile
import tempfile
import shutil

# I/O layout (unchanged)
OM_DIR = os.path.join("data", "om_baseline")
OUT_DIR = os.path.join("data", "train_tiles")
os.makedirs(OUT_DIR, exist_ok=True)

log = logging.getLogger("make_pairs_plus")

# --- S3 helpers for mirroring parquet files ---
try:
    from utils.s3_utils import s3_enabled, upload_file, list_objects
except Exception:
    try:
        from s3_utils import s3_enabled, upload_file, list_objects  # type: ignore
    except Exception:

        def s3_enabled() -> bool:
            return False

        def upload_file(*args, **kwargs):
            return None
        
        def list_objects(*args, **kwargs):
            return []


# -------------------------
# Logging / small utils
# -------------------------
def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
# Manifest (already-processed OM files per tile)
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
# Robust lat/lon parsing (unchanged API, extended paths)
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

    def _from_descriptor(desc):
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
                return float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
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
        return list(np.linspace(lat_min, lat_max, ny)), list(
            np.linspace(lon_min, lon_max, nx)
        )

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


def _parse_times_from_hourly_block(block):
    if not isinstance(block, dict) or "time" not in block:
        return None
    try:
        return [dt.datetime.fromisoformat(t.replace("Z", "")) for t in block["time"]]
    except Exception:
        return None


def _extract_times(j):
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
        return t if t else None
    if isinstance(om, list) and om:
        hb = om[0].get("hourly", {})
        t = _parse_times_from_hourly_block(hb)
        return t if t else None
    return None


def _lon_canon(l):
    x = ((float(l) + 180.0) % 360.0) - 180.0
    return -180.0 if abs(x - 180.0) < 1e-9 else x


def _lon_diff_deg(a, b):
    A = _lon_canon(a)
    B = _lon_canon(b)
    d = ((A - B + 540.0) % 360.0) - 180.0
    return -180.0 if abs(d - 180.0) < 1e-9 else d


def _nearest_idx_points(points_lat, points_lon, req_lat, req_lon):
    lat_arr = np.asarray(points_lat, dtype=float)
    lon_arr = np.asarray(points_lon, dtype=float)
    dlat = lat_arr - float(req_lat)
    dlon = np.array([_lon_diff_deg(l, req_lon) for l in lon_arr], dtype=float)
    w = np.cos(np.deg2rad(float(req_lat)))
    dist2 = dlat * dlat + (w * dlon) * (w * dlon)
    return int(np.argmin(dist2)), float(np.min(dist2))


def _build_matrix_from_points(point_list, name, times, grid, tol_deg=0.6):
    T, G = len(times), len(grid)
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
        if p_times and len(p_times) == v.size and v.size != T:
            aligned = np.empty(T, dtype=float)
            for i, tt in enumerate(times):
                j = int(np.argmin([abs((tt - pt).total_seconds()) for pt in p_times]))
                aligned[i] = float(v[j])
            v = aligned
        elif v.size != T:
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
        if abs(la_req) >= 89.5:
            dlat = np.abs(pts_lat - la_req)
            idx = int(np.argmin(dlat))
            d2 = float(dlat[idx] ** 2)
        else:
            dlat = pts_lat - la_req
            dlon = np.array([_lon_diff_deg(L, lo_req) for L in pts_lon], dtype=float)
            w = np.cos(np.deg2rad(la_req))
            dist2 = dlat * dlat + (w * dlon) * (w * dlon)
            idx = int(np.argmin(dist2))
            d2 = float(dist2[idx])
        if d2 <= tol2:
            out[:, gi] = series[idx]
    if np.all(~np.isfinite(out)):
        raise KeyError(f"no points matched requested grid for var {name}")
    return out


def _parse_baseline(j, key, times, grid, required_vars):
    blk = j.get(key, None)
    if blk is None:
        return {v: None for v in required_vars}
    T, G = len(times), len(grid)
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
# ERA5 resolution + sampling (hybrid + nearest)
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
    # Land extras:
    "skt": ["skt", "skin_temperature"],
    "sde": ["sde", "snow_depth"],
    "swvl1": ["swvl1", "volumetric_soil_water_layer_1"],
    "swvl2": ["swvl2", "volumetric_soil_water_layer_2"],
    "swvl3": ["swvl3", "volumetric_soil_water_layer_3"],
    "swvl4": ["swvl4", "volumetric_soil_water_layer_4"],
}


def _find_var_name(ds: xr.Dataset, key: str) -> str:
    aliases = VAR_ALIASES.get(key, [key])
    data_vars_lower = {name.lower(): name for name in ds.data_vars}
    for cand in aliases:
        name = data_vars_lower.get(cand.lower())
        if name is not None:
            return name
    coords_lower = {name.lower(): name for name in ds.coords}
    for cand in aliases:
        name = coords_lower.get(cand.lower())
        if name is not None:
            return name
    raise KeyError(
        f"No variable named {aliases!r}. Variables include {list(ds.data_vars)}"
    )


def _maybe_squeeze_members(da: xr.DataArray) -> xr.DataArray:
    for extra in ("expver", "number"):
        if extra in da.dims:
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


def _has_var(ds, key):
    try:
        _find_var_name(ds, key)
        return True
    except KeyError:
        return False


def _dims_for(da: xr.DataArray):
    lat_dim = next((d for d in ("latitude", "lat", "y") if d in da.dims), None)
    lon_dim = next((d for d in ("longitude", "lon", "x") if d in da.dims), None)
    time_dim = next((d for d in ("time", "valid_time", "t") if d in da.dims), None)
    if not lat_dim or not lon_dim or not time_dim:
        raise KeyError(f"Could not resolve dims for dataarray; dims={da.dims}")
    return lat_dim, lon_dim, time_dim


def _index_nearest(vals: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(vals, dtype=float) - float(target))))


def _map_target_lon_for_coords(lon_coords: np.ndarray, lo_req: float) -> float:
    lon_coords = np.asarray(lon_coords, dtype=float)
    if np.nanmin(lon_coords) >= 0.0 and np.nanmax(lon_coords) > 180.0:
        x = (float(lo_req) + 360.0) % 360.0
        return 0.0 if abs(x - 360.0) < 1e-9 else x
    x = ((float(lo_req) + 180.0) % 360.0) - 180.0
    return -180.0 if abs(x - 180.0) < 1e-9 else x


def nearest_truth(ds, key, grid, times):
    vname = _find_var_name(ds, key)
    da = _maybe_squeeze_members(ds[vname])
    lat_dim, lon_dim, time_dim = _dims_for(da)
    lats = _coord_values(ds, ("latitude", "lat", "y"))
    lons = _coord_values(ds, ("longitude", "lon", "x"))
    tlist = _time_list(ds)
    t_idx = [
        int(np.argmin([abs((tt - t0).total_seconds()) for t0 in tlist])) for tt in times
    ]
    T, G = len(times), len(grid)
    out = np.empty((T, G), dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    for gi, (la_req, lo_req) in enumerate(grid):
        la_req = float(la_req)
        if abs(la_req) >= 89.5:
            ilat = _index_nearest(lats, la_req)
            ilon = 0
        else:
            ilat = _index_nearest(lats, la_req)
            lo_adj = _map_target_lon_for_coords(lons, lo_req)
            ilon = _index_nearest(lons, lo_adj)
        out[:, gi] = da.isel(
            {lat_dim: ilat, lon_dim: ilon, time_dim: xr.DataArray(t_idx)}
        ).values
    return out


def hybrid_truth(era5_land: xr.Dataset, era5_single: xr.Dataset, key: str, grid, times):
    try_land = _has_var(era5_land, key)
    try_single = _has_var(era5_single, key)
    if not try_single and not try_land:
        raise KeyError(f"No dataset provides variable '{key}'")
    land_arr = nearest_truth(era5_land, key, grid, times) if try_land else None
    single_arr = nearest_truth(era5_single, key, grid, times) if try_single else None
    if land_arr is None:
        return single_arr
    if single_arr is None:
        return land_arr
    land_ok = np.isfinite(land_arr)
    return np.where(land_ok, land_arr, single_arr)


# -------------------------
# ERA5 manifest + S3 helpers
# -------------------------
def _load_json_uri(uri: str):
    """Load a small JSON file from local disk or a remote URI (e.g. s3://...)."""
    if isinstance(uri, str) and uri.startswith("s3://"):
        # Use boto3 via s3_utils instead of fsspec (which requires s3fs)
        try:
            from utils.s3_utils import _bucket_and_prefix, _client, _join_key
        except Exception:
            from s3_utils import _bucket_and_prefix, _client, _join_key  # type: ignore
        
        # Parse s3://bucket/key into bucket and key
        parts = uri.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI: {uri}")
        bucket_from_uri, key_from_uri = parts
        
        # Use boto3 to download the JSON file
        cli = _client()
        try:
            resp = cli.get_object(Bucket=bucket_from_uri, Key=key_from_uri)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load manifest from S3: {uri} - {e}")
    elif isinstance(uri, str) and "://" in uri:
        # Fallback to fsspec for other protocols (http, https, etc.)
        try:
            with fsspec.open(uri, "rt") as f:
                return json.load(f)
        except ImportError:
            raise ImportError(f"fsspec with appropriate backend required for URI: {uri}")
    else:
        # Local file
        with open(uri, "r") as f:
            return json.load(f)


def _uris_from_manifest(meta: dict) -> list[str]:
    """Turn manifest['parts'][*]['key'] into a list of URIs.

    If keys are plain paths, WEATHER_S3_BUCKET / WEATHER_S3_PREFIX are used to
    construct s3:// URLs; otherwise they are returned as-is.
    """
    parts = meta.get("parts") or []
    keys: list[str] = []
    for p in parts:
        k = (p or {}).get("key")
        if k:
            keys.append(str(k))

    uris: list[str] = []
    bucket = os.environ.get("WEATHER_S3_BUCKET")
    prefix = os.environ.get("WEATHER_S3_PREFIX", "").strip("/")
    for k in keys:
        if k.startswith("s3://"):
            uris.append(k)
            continue
        if bucket:
            # Normalise relative keys under PREFIX when present.
            if prefix and not k.startswith(prefix + "/"):
                k_path = f"{prefix}/{k.lstrip('/')}"
            else:
                k_path = k.lstrip("/")
            uris.append(f"s3://{bucket}/{k_path}")
        else:
            # Fallback: treat as local path (relative or absolute)
            uris.append(k)
    return uris


def _open_era5_from_manifest(manifest_uri: str) -> xr.Dataset:
    """Open an ERA5 dataset by streaming all chunk files listed in a manifest."""
    meta = _load_json_uri(manifest_uri)
    uris = _uris_from_manifest(meta)
    if not uris:
        raise SystemExit(f"Manifest {manifest_uri!r} contained no 'parts'")
    log.info(f"[era5] opening {len(uris)} chunks from manifest {manifest_uri}")
    
    # Check if any URIs are S3 URIs
    has_s3 = any(uri.startswith("s3://") for uri in uris)
    
    if has_s3:
        # For S3 URIs, download to temp directory and open locally
        # This is more reliable than trying to stream from S3 with netCDF4
        try:
            import s3fs
            import tempfile
            import shutil
            import zipfile
        except ImportError:
            raise ImportError(
                "s3fs is required for reading S3 files. Install with: pip install s3fs"
            )
        
        fs = s3fs.S3FileSystem()
        temp_dir = tempfile.mkdtemp(prefix="era5_temp_")
        
        try:
            log.info(f"[era5] downloading {len(uris)} files from S3 to temp directory")
            local_paths = []
            
            for i, uri in enumerate(uris):
                # Determine local filename
                base_name = os.path.basename(uri)
                # Remove .nc extension if present, we'll detect the actual format
                if base_name.endswith(".nc"):
                    base_name = base_name[:-3]
                local_path = os.path.join(temp_dir, f"chunk_{i}_{base_name}")
                
                # Download from S3
                fs.get(uri, local_path)
                
                # Check if it's a ZIP file and extract if needed
                try:
                    with open(local_path, "rb") as f:
                        header = f.read(4)
                    if header.startswith(b"PK\x03\x04"):
                        # It's a ZIP file, extract it
                        log.debug(f"[era5] extracting ZIP file: {base_name}")
                        with zipfile.ZipFile(local_path, "r") as zf:
                            # Extract first file (should be the NetCDF)
                            extracted = zf.namelist()[0]
                            zf.extract(extracted, temp_dir)
                            extracted_path = os.path.join(temp_dir, extracted)
                            # Remove zip file
                            os.remove(local_path)
                            local_path = extracted_path
                except Exception as e:
                    log.debug(f"[era5] not a ZIP file or extraction failed: {e}")
                
                local_paths.append(local_path)
            
            # Open local files - try h5netcdf first, then netcdf4
            log.info(f"[era5] opening {len(local_paths)} local files")
            try:
                ds = xr.open_mfdataset(
                    local_paths,
                    combine="by_coords",
                    engine="h5netcdf"
                )
            except Exception as e:
                log.warning(f"[era5] h5netcdf failed, trying netcdf4: {e}")
                ds = xr.open_mfdataset(
                    local_paths,
                    combine="by_coords",
                    engine="netcdf4"
                )
            
            # Store temp_dir in dataset attributes for cleanup later
            ds.attrs["_temp_dir"] = temp_dir
            log.info(f"[era5] dataset opened, temp files in {temp_dir} (will be cleaned up on close)")
            return ds
    else:
        # Local files - use standard approach
        return xr.open_mfdataset(uris, combine="by_coords")


# -------------------------
# Feature engineering helpers (new)
# -------------------------
def _lags(A: np.ndarray, ks=(1, 3, 6, 24)) -> dict:
    # A: (T,G)
    feats = {}
    for k in ks:
        lag = np.full_like(A, np.nan, dtype=float)
        lag[k:, :] = A[:-k, :]
        feats[f"lag{k}"] = lag
    return feats


def _local_stats(A: np.ndarray, nlat: int, nlon: int, radius=1):
    # A: (T,G) in row-major flattening with lat major, lon minor
    out_mean = np.full_like(A, np.nan, dtype=float)
    out_grad = np.full_like(A, np.nan, dtype=float)
    for ti in range(A.shape[0]):
        M = A[ti].reshape(nlat, nlon)
        # 3x3 mean (naive conv)
        pad = np.pad(M, radius, mode="edge")
        acc = np.zeros_like(M)
        cnt = (2 * radius + 1) ** 2
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                acc += pad[
                    radius + di : radius + di + nlat, radius + dj : radius + dj + nlon
                ]
        mean3 = acc / cnt
        # gradient magnitude
        gx = np.zeros_like(M)
        gy = np.zeros_like(M)
        gx[:, 1:-1] = (M[:, 2:] - M[:, :-2]) * 0.5
        gy[1:-1, :] = (M[2:, :] - M[:-2, :]) * 0.5
        grad = np.hypot(gx, gy)
        out_mean[ti] = mean3.reshape(-1)
        out_grad[ti] = grad.reshape(-1)
    return out_mean, out_grad


def _unique_lat_lon_counts(grid):
    lats = [la for la, _ in grid]
    lons = [lo for _, lo in grid]
    nlat = len(sorted(set([round(float(x), 6) for x in lats])))
    nlon = len(sorted(set([round(float(x), 6) for x in lons])))
    return nlat if nlat > 0 else 1, nlon if nlon > 0 else max(1, len(grid))


# -------------------------
# Robust parquet writer (append+dedupe)
# -------------------------
def safe_to_parquet(
    df: pd.DataFrame, path: str, append: bool = False, subset=("lat", "lon", "time")
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _write(df_final: pd.DataFrame):
        engines = []
        try:
            import fastparquet  # noqa

            engines.append(("fastparquet", {}))
        except Exception:
            pass
        last_err = None
        for eng, kwargs in engines:
            try:
                log.debug(f"[parquet] writing {path} using engine={eng}")
                df_final.to_parquet(path, engine=eng, index=False, **kwargs)
                # Mirror to S3 (optional)
                if s3_enabled():
                    try:
                        upload_file(path, subdir="train_tiles", key=None)
                        log.info(
                            f"[parquet] mirrored to S3: train_tiles/{os.path.basename(path)}"
                        )
                        # In S3-only mode we don't keep a local training file copy.
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                    except Exception as e:
                        log.warning(f"[parquet] S3 upload failed for {path}: {e!r}")
                return
            except Exception as e:
                log.warning(f"[parquet] engine={eng} failed: {e!r}")
                last_err = e
        # Fallback: CSV
        csv_path = os.path.splitext(path)[0] + ".csv"
        df_final.to_csv(csv_path, index=False)
        log.error(
            f"[parquet] all parquet engines failed; wrote CSV fallback: {csv_path}"
        )
        if s3_enabled():
            try:
                upload_file(csv_path, subdir="train_tiles", key=None)
                log.info(
                    f"[parquet] mirrored CSV to S3: train_tiles/{os.path.basename(csv_path)}"
                )
                try:
                    os.remove(csv_path)
                except Exception:
                    pass
            except Exception as e:
                log.warning(f"[parquet] S3 upload failed for {csv_path}: {e!r}")
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
    ap.add_argument(
        "--era5-single",
        required=False,
        help="Path to merged ERA5 single-level NetCDF (local or s3://)",
    )
    ap.add_argument(
        "--era5-land",
        required=False,
        help="Path to merged ERA5-Land NetCDF (local or s3://)",
    )
    ap.add_argument(
        "--era5-single-manifest",
        required=False,
        help="Manifest JSON (local or s3://) listing ERA5 single-level chunk files",
    )
    ap.add_argument(
        "--era5-land-manifest",
        required=False,
        help="Manifest JSON (local or s3://) listing ERA5-Land chunk files",
    )
    ap.add_argument("--tile-id", required=True)
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore manifest and rebuild from all OM JSONs",
    )
    args = ap.parse_args()

    # Require ERA5 inputs either via direct NetCDF paths or via S3 manifest(s).
    if not args.era5_single and not args.era5_single_manifest:
        ap.error("one of --era5-single or --era5-single-manifest is required")
    if not args.era5_land and not args.era5_land_manifest:
        ap.error("one of --era5-land or --era5-land-manifest is required")

    setup_logging(args.log_level)

    # Track temp directories for cleanup
    temp_dirs_to_cleanup = []
    
    # Load ERA5 single-levels
    if args.era5_single_manifest:
        log.info(f"Loading ERA5 Single from manifest: {args.era5_single_manifest}")
        era5_single = _open_era5_from_manifest(args.era5_single_manifest)
        if "_temp_dir" in era5_single.attrs:
            temp_dirs_to_cleanup.append(era5_single.attrs["_temp_dir"])
    else:
        log.info(f"Loading ERA5 Single: {args.era5_single}")
        era5_single = xr.open_dataset(args.era5_single)

    # Load ERA5-Land
    if args.era5_land_manifest:
        log.info(f"Loading ERA5-Land from manifest: {args.era5_land_manifest}")
        era5_land = _open_era5_from_manifest(args.era5_land_manifest)
        if "_temp_dir" in era5_land.attrs:
            temp_dirs_to_cleanup.append(era5_land.attrs["_temp_dir"])
    else:
        log.info(f"Loading ERA5-Land:  {args.era5_land}")
        era5_land = xr.open_dataset(args.era5_land)

    # Rows per target (same keys as before)
    rows = {k: [] for k in ["t2m", "td2m", "psfc", "tp", "wspd100", "wdir100"]}

    # Collect OM files from local directory and/or S3
    files = []
    file_sources = {}  # Map filename -> (source_type, path_or_uri)
    
    # First, check local directory
    local_files = sorted(glob.glob(os.path.join(OM_DIR, "omifs_*.json")))
    for p in local_files:
        base = os.path.basename(p)
        files.append(p)
        file_sources[base] = ("local", p)
    
    # Then, check S3 if enabled
    if s3_enabled():
        try:
            s3_files = list_objects("om_baseline", prefix="omifs_", suffix=".json")
            for s3_uri in s3_files:
                name = os.path.basename(s3_uri)
                # Only add if not already in local files
                if name not in file_sources:
                    files.append(s3_uri)
                    file_sources[name] = ("s3", s3_uri)
        except Exception as e:
            log.warning(f"[pair] could not list S3 OM files: {e}, using local only")
    
    if not files:
        raise SystemExit("No OM baseline files found (checked local and S3)")
    
    log.info(f"[pair] found {len(files)} OM files ({len([f for f in file_sources.values() if f[0]=='s3'])} from S3, {len([f for f in file_sources.values() if f[0]=='local'])} local)")

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
            # Load from local file or S3 URI
            if isinstance(p, str) and (p.startswith("s3://") or "://" in p):
                j = _load_json_uri(p)
            else:
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

        nlat, nlon = _unique_lat_lon_counts(grid)
        log.info(f"[pair] {base}: T={T}, G={G} (nlat={nlat}, nlon={nlon})")

        t0 = times[0]
        leads = np.array(
            [(t - t0).total_seconds() / 3600.0 for t in times], dtype=float
        )
        hods = np.array([t.hour + t.minute / 60 for t in times], dtype=float)
        doys = np.array([t.timetuple().tm_yday for t in times], dtype=float)
        hod_sin = np.sin(2 * np.pi * hods / 24.0)
        hod_cos = np.cos(2 * np.pi * hods / 24.0)
        doy_sin = np.sin(2 * np.pi * doys / 365.25)
        doy_cos = np.cos(2 * np.pi * doys / 365.25)

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

        # ERA5 truth sampling (hybrid where available)
        try:
            t2m_truth = to_celsius(
                hybrid_truth(era5_land, era5_single, "t2m", grid, times)
            )
            td2m_truth = to_celsius(
                hybrid_truth(era5_land, era5_single, "d2m", grid, times)
            )
            tp_truth = m_to_mm(hybrid_truth(era5_land, era5_single, "tp", grid, times))
            sp_truth = pa_to_hpa(
                hybrid_truth(era5_land, era5_single, "sp", grid, times)
            )
            u100 = nearest_truth(era5_single, "u100", grid, times)
            v100 = nearest_truth(era5_single, "v100", grid, times)
            wspd_truth = ms_to_kmh(np.hypot(u100, v100))
            wdir_truth = (np.degrees(np.arctan2(-u100, -v100)) + 360.0) % 360.0

            # ERA5-Land exogenous predictors (used as features)
            skt = hybrid_truth(era5_land, era5_single, "skt", grid, times)  # K
            sde = hybrid_truth(era5_land, era5_single, "sde", grid, times)  # m
            swvl1 = hybrid_truth(era5_land, era5_single, "swvl1", grid, times)
            swvl2 = hybrid_truth(era5_land, era5_single, "swvl2", grid, times)
            swvl3 = hybrid_truth(era5_land, era5_single, "swvl3", grid, times)
            swvl4 = hybrid_truth(era5_land, era5_single, "swvl4", grid, times)
        except Exception as e:
            log.warning(f"[pair] skip {base} (ERA5 sample error: {e})")
            continue

        # Derived baseline transforms (shared)
        TminusTd = b_t2m_om - b_td2m_om
        wdir_rad = np.deg2rad(b_wdir_om)
        u_om = -b_wspd_om * np.sin(wdir_rad)
        v_om = -b_wspd_om * np.cos(wdir_rad)
        dspfc_3h = np.full_like(b_psfc_om, np.nan)
        dspfc_3h[3:, :] = b_psfc_om[3:, :] - b_psfc_om[:-3, :]

        # Spatial stats on a representative baseline (t2m) for local context
        t2m_mean3, t2m_grad3 = _local_stats(b_t2m_om, nlat, nlon, radius=1)

        # Variable-specific lag bundles (baseline_om)
        lags = {
            "t2m": _lags(b_t2m_om),
            "td2m": _lags(b_td2m_om),
            "psfc": _lags(b_psfc_om),
            "tp": _lags(b_tp_om),
            "wspd100": _lags(b_wspd_om),
            "wdir100": _lags(b_wdir_om),
        }

        def emit(rows_list, bom, bifs, truth, var_key: str):
            for gi, (la, lo) in enumerate(grid):
                for ti, tt in enumerate(times):
                    bomv = float(bom[ti, gi])
                    bifsv = (
                        float(bifs[ti, gi])
                        if (bifs is not None and np.isfinite(bifs[ti, gi]))
                        else np.nan
                    )
                    row = {
                        # core features (backward compatible)
                        "lat": la,
                        "lon": lo,
                        "hod": float(hods[ti]),
                        "lead": float(ti),
                        "baseline_om": bomv,
                        "baseline_ifs": bifsv,
                        "target": float(truth[ti, gi]),
                        "time": times[ti].isoformat(),
                        # cycles
                        "hod_sin": float(hod_sin[ti]),
                        "hod_cos": float(hod_cos[ti]),
                        "doy_sin": float(doy_sin[ti]),
                        "doy_cos": float(doy_cos[ti]),
                        # transforms
                        "tminus_td": float(TminusTd[ti, gi]),
                        "u100_om": float(u_om[ti, gi]),
                        "v100_om": float(v_om[ti, gi]),
                        "dspfc_3h": float(dspfc_3h[ti, gi]),
                        # simple spatial context
                        "t2m_mean3": float(t2m_mean3[ti, gi]),
                        "t2m_grad3": float(t2m_grad3[ti, gi]),
                        # land exogenous (converted where helpful)
                        "skin_temp": float(to_celsius(skt[ti, gi])),
                        "snow_depth": float(sde[ti, gi]),
                        "swvl1": float(swvl1[ti, gi]),
                        "swvl2": float(swvl2[ti, gi]),
                        "swvl3": float(swvl3[ti, gi]),
                        "swvl4": float(swvl4[ti, gi]),
                    }
                    # lags for the current variable's baseline_om
                    for k, A in lags[var_key].items():
                        row[k] = float(A[ti, gi])
                    rows_list.append(row)

        emit(rows["t2m"], b_t2m_om, b_t2m_ifs, t2m_truth, "t2m")
        emit(rows["td2m"], b_td2m_om, b_td2m_ifs, td2m_truth, "td2m")
        emit(rows["psfc"], b_psfc_om, b_psfc_ifs, sp_truth, "psfc")
        emit(rows["tp"], b_tp_om, b_tp_ifs, tp_truth, "tp")
        emit(rows["wspd100"], b_wspd_om, b_wspd_ifs, wspd_truth, "wspd100")
        emit(rows["wdir100"], b_wdir_om, b_wdir_ifs, wdir_truth, "wdir100")

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
    
    # Clean up temp directories used for S3 downloads
    for temp_dir in temp_dirs_to_cleanup:
        try:
            if os.path.exists(temp_dir):
                log.info(f"[cleanup] removing temp directory: {temp_dir}")
                shutil.rmtree(temp_dir)
        except Exception as e:
            log.warning(f"[cleanup] failed to remove {temp_dir}: {e}")


if __name__ == "__main__":
    main()
