#!/usr/bin/env python
import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from weather_plus.config import MODEL_DIR

df = pd.read_parquet("data/train/t2m_training.parquet")
X = df[
    ["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs", "baseline_diff"]
].values
y = df["target"].values
m = Ridge(alpha=0.8).fit(X, y)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(m, os.path.join(MODEL_DIR, "temperature_2m.joblib"))
print("Saved temperature_2m.joblib")
