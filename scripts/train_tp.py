#!/usr/bin/env python
import os, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from weather_plus.config import MODEL_DIR

df = pd.read_parquet("data/train/tp_training.parquet")
Xb = df[["lat", "lon", "hod", "lead", "baseline_om", "baseline_ifs"]].values
y = df["target"].values
is_rain = (y > 0.05).astype(int)

clf = LogisticRegression(max_iter=600).fit(Xb, is_rain)

mask = is_rain == 1
X_amt = Xb[mask]
y_amt = np.log1p(np.clip(y[mask], 0, None))
gbr = GradientBoostingRegressor(
    loss="huber", alpha=0.9, n_estimators=400, max_depth=3, learning_rate=0.05
)
gbr.fit(X_amt, y_amt)


class PrecipModel:
    def __init__(self, clf, gbr):
        self.clf, self.gbr = clf, gbr

    def predict(self, X):
        pr = self.clf.predict_proba(X)[:, 1]
        amt = np.expm1(self.gbr.predict(X))
        return np.clip(pr * np.clip(amt, 0, None), 0, None)

    def __getstate__(self):
        return {"clf": self.clf, "gbr": self.gbr}

    def __setstate__(self, s):
        self.clf, self.gbr = s["clf"], s["gbr"]


art = PrecipModel(clf, gbr)
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(art, os.path.join(MODEL_DIR, "precipitation.joblib"))
print("Saved precipitation.joblib")
