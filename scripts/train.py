#!/usr/bin/env python
import os, glob, json, joblib, numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from weather_plus.config import MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
PT = "data/train_tiles"

# Variables we produce (same names your pairing & miner expect)
TASKS = [
    ("t2m", "temperature_2m", "reg"),
    ("td2m", "dew_point_2m", "reg"),
    ("psfc", "surface_pressure", "reg"),
    ("tp", "precipitation", "tp2stage"),
    ("wspd100", "wind_speed_100m", "wspd"),
    ("wdir100", "wind_direction_100m", "wdir"),
]

# Base features (old) + suggested new ones – safely subset on availability
BASE = ["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs"]
EXTRA = [
    "hod_sin",
    "hod_cos",
    "doy_sin",
    "doy_cos",
    "tminus_td",
    "u100_om",
    "v100_om",
    "dspfc_3h",
    "t2m_mean3",
    "t2m_grad3",
    "skin_temp",
    "snow_depth",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "lag1",
    "lag3",
    "lag6",
    "lag24",
]
PREF = BASE + EXTRA


@dataclass
class Bundle:
    model: object
    feature_names: list
    meta: dict


def _select_X(df, pref=PREF):
    cols = [c for c in pref if c in df.columns]
    return df[cols].values, cols


def fit_reg(df):
    X, cols = _select_X(df)
    y = df["target"].values
    m = HistGradientBoostingRegressor(
        max_leaf_nodes=31,
        learning_rate=0.06,
        max_depth=None,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
    )
    m.fit(X, y)
    return Bundle(m, cols, {"task": "reg"})


def fit_tp_two_stage(df):
    X, cols = _select_X(df)
    y = df["target"].values
    wet = (y > 0.0).astype(int)
    clf = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06)
    clf.fit(X, wet)
    # regress on wet-only, log1p
    mask = wet == 1
    reg = HistGradientBoostingRegressor(max_leaf_nodes=31, learning_rate=0.06)
    reg.fit(X[mask], np.log1p(y[mask]))
    return Bundle((clf, reg), cols, {"task": "tp2stage"})


def fit_wspd(df):
    # sqrt trick + HGBR
    df = df.copy()
    df["tgt"] = np.sqrt(np.clip(df["target"].values, 0, None))
    X, cols = _select_X(df)
    m = HistGradientBoostingRegressor(max_leaf_nodes=31, learning_rate=0.06)
    m.fit(X, df["tgt"].values)
    return Bundle(m, cols, {"task": "wspd"})


def fit_wdir(df):
    th = np.deg2rad(df["target"].values)
    X, cols = _select_X(df)
    ms = HistGradientBoostingRegressor(max_leaf_nodes=31, learning_rate=0.06)
    mc = HistGradientBoostingRegressor(max_leaf_nodes=31, learning_rate=0.06)
    ms.fit(X, np.sin(th))
    mc.fit(X, np.cos(th))
    return Bundle((ms, mc), cols, {"task": "wdir"})


FIT = {"reg": fit_reg, "tp2stage": fit_tp_two_stage, "wspd": fit_wspd, "wdir": fit_wdir}


def _save(bundle, out):
    joblib.dump(bundle, out)


tiles = sorted(
    set(
        os.path.basename(p).split("__")[0]
        for p in glob.glob(os.path.join(PT, "*__*.parquet"))
    )
)

for tile in tiles:
    print("=== Tile", tile)
    for key, var, kind in TASKS:
        path = os.path.join(PT, f"{tile}__{key}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        bundle = FIT[kind](df)
        _save(bundle, os.path.join(MODEL_DIR, f"{var}__{tile}.joblib"))
        print("saved", var, tile, "with", len(bundle.feature_names), "features")

# Global fallback
for key, var, kind in TASKS:
    parts = glob.glob(os.path.join(PT, f"*__{key}.parquet"))
    if not parts:
        continue
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if df.empty:
        continue
    bundle = FIT[kind](df)
    _save(bundle, os.path.join(MODEL_DIR, f"{var}.joblib"))
    print("saved global", var, "with", len(bundle.feature_names), "features")
