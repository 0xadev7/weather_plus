#!/usr/bin/env python
import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from weather_plus.config import MODEL_DIR

df = pd.read_parquet("data/train/wspd100_training.parquet")
b = np.sqrt(np.clip(df["baseline_om"].values, 0, None))
b2 = np.sqrt(
    np.clip(
        np.nan_to_num(df["baseline_ifs"].values, nan=df["baseline_om"].values), 0, None
    )
)
X = df[["lat", "lon", "hod", "lead"]].values
X = np.column_stack([X, b, b2, (b - b2)])
y = np.sqrt(np.clip(df["target"].values, 0, None))
m = Ridge(alpha=0.7).fit(X, y)


class WspdModel:
    def __init__(self, inner):
        self.inner = inner

    def predict(self, Xfull):
        base1 = np.sqrt(np.clip(Xfull[:, 4], 0, None))
        base2 = np.sqrt(np.clip(np.nan_to_num(Xfull[:, 5], nan=Xfull[:, 4]), 0, None))
        Z = np.column_stack([Xfull[:, 0:4], base1, base2])
        y = self.inner.predict(Z)
        return np.clip(y**2, 0, None)

    def __getstate__(self):
        return {"inner": self.inner}

    def __setstate__(self, s):
        self.inner = s["inner"]


art = WspdModel(m)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(art, os.path.join(MODEL_DIR, "wind_speed_100m.joblib"))
print("Saved wind_speed_100m.joblib")
