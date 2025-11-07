#!/usr/bin/env python
import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from weather_plus.config import MODEL_DIR

df = pd.read_parquet("data/train/psfc_training.parquet")
X = df[
    ["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs", "baseline_diff"]
].values
y = df["target"].values
m = Ridge(alpha=0.5).fit(X, y)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(m, os.path.join(MODEL_DIR, "surface_pressure.joblib"))
print("Saved surface_pressure.joblib")
