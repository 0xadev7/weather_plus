#!/usr/bin/env python
import os, glob, joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from weather_plus.config import MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
PT = "data/train_tiles"


def train_linear(df):
    X = df[["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs"]].values
    y = df["target"].values
    return Ridge(alpha=0.8).fit(X, y)


def train_wspd(df):
    b1 = np.sqrt(np.clip(df["baseline_om"].values, 0, None))
    b2 = np.sqrt(
        np.clip(
            np.nan_to_num(df["baseline_ifs"].values, nan=df["baseline_om"].values),
            0,
            None,
        )
    )
    X = np.column_stack([df[["lat", "lon", "hod", "lead"]].values, b1, b2])
    y = np.sqrt(np.clip(df["target"].values, 0, None))
    m = Ridge(alpha=0.7).fit(X, y)

    class Wrap:
        def __init__(self, inner):
            self.inner = inner

        def predict(self, Xfull):
            b1 = np.sqrt(np.clip(Xfull[:, 4], 0, None))
            b2 = np.sqrt(np.clip(np.nan_to_num(Xfull[:, 5], nan=Xfull[:, 4]), 0, None))
            Z = np.column_stack([Xfull[:, 0:4], b1, b2])
            y = self.inner.predict(Z)
            return np.clip(y**2, 0, None)

    return Wrap(m)


def train_wdir(df):
    X = df[["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs"]].values
    th = np.deg2rad(df["target"].values)
    s, c = np.sin(th), np.cos(th)
    ms = Ridge(alpha=0.8).fit(X, s)
    mc = Ridge(alpha=0.8).fit(X, c)

    class Wrap:
        def __init__(self, ms, mc):
            self.ms, self.mc = ms, mc

        def predict(self, X):
            s = self.ms.predict(X)
            c = self.mc.predict(X)
            return (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0

    return Wrap(ms, mc)


TASKS = [
    ("t2m", "temperature_2m", train_linear),
    ("td2m", "dew_point_2m", train_linear),
    ("psfc", "surface_pressure", train_linear),
    ("tp", "precipitation", train_linear),  # swap in your 2-stage model if desired
    ("wspd100", "wind_speed_100m", train_wspd),
    ("wdir100", "wind_direction_100m", train_wdir),
]

tiles = sorted(
    set(
        [
            os.path.basename(p).split("__")[0]
            for p in glob.glob(os.path.join(PT, "*__*.parquet"))
        ]
    )
)

for tile in tiles:
    print("=== Tile", tile)
    for key, var, trainer in TASKS:
        path = os.path.join(PT, f"{tile}__{key}.parquet")
        if not os.path.exists(path):
            print("skip", var, "(no data)")
            continue
        df = pd.read_parquet(path)
        if df.empty:
            print("skip", var, "(empty)")
            continue
        m = trainer(df)
        joblib.dump(m, os.path.join(MODEL_DIR, f"{var}__{tile}.joblib"))
        print("saved", var, tile)

# Optional: also train global models from concatenating all tiles (for fallback)
for key, var, trainer in TASKS:
    parts = glob.glob(os.path.join(PT, f"*__{key}.parquet"))
    if not parts:
        continue
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if df.empty:
        continue
    m = trainer(df)
    joblib.dump(m, os.path.join(MODEL_DIR, f"{var}.joblib"))
    print("saved global", var)
