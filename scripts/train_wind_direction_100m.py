#!/usr/bin/env python
import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from weather_plus.config import MODEL_DIR

df = pd.read_parquet("data/train/wdir100_training.parquet")
X = df[
    ["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs", "baseline_diff"]
].values
theta = np.deg2rad(df["target"].values)
y_sin = np.sin(theta)
y_cos = np.cos(theta)
m_sin = Ridge(alpha=0.8).fit(X, y_sin)
m_cos = Ridge(alpha=0.8).fit(X, y_cos)


class WdirModel:
    def __init__(self, ms, mc):
        self.ms, self.mc = ms, mc

    def predict(self, X):
        s = self.ms.predict(X)
        c = self.mc.predict(X)
        ang = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
        return ang

    def __getstate__(self):
        return {"ms": self.ms, "mc": self.mc}

    def __setstate__(self, s):
        self.ms, self.mc = s["ms"], s["mc"]


art = WdirModel(m_sin, m_cos)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(art, os.path.join(MODEL_DIR, "wind_direction_100m.joblib"))
print("Saved wind_direction_100m.joblib")
