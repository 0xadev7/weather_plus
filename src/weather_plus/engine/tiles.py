from __future__ import annotations
from typing import Tuple

# 30-degree bands → 6 lat bands * 12 lon bands = 72 tiles
LAT_BANDS = [-90, -60, -30, 0, 30, 60, 90]
LON_BANDS = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]


def _band_index(x, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2  # include the last edge case


def tile_id(lat: float, lon: float) -> str:
    i = _band_index(lat, LAT_BANDS)
    j = _band_index(lon, LON_BANDS)
    return f"LAT{i}_LON{j}"


def artifact_name(var: str, tile: str) -> str:
    # e.g., "temperature_2m__LAT3_LON7.joblib"
    return f"{var}__{tile}.joblib"
