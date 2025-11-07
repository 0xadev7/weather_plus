from __future__ import annotations
import os
from typing import Optional, Dict, Any
import numpy as np
from joblib import load


class Calibrator:
    """
    Loads per-variable calibration artifacts (e.g., EMOS/NGR linear/GAM/GBDT/QRF).
    If no artifact is present, falls back to identity (or simple bias correction
    with stored climatology if available).
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.cache: Dict[str, Any] = {}

    def _artifact_path(self, var: str) -> str:
        return os.path.join(self.model_dir, f"{var}.joblib")

    def _load(self, var: str):
        path = self._artifact_path(var)
        if os.path.exists(path):
            self.cache[var] = load(path)
        else:
            self.cache[var] = None

    def predict(self, var: str, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """
        X: engineered features [n, f] (can be minimal: baseline, lead, hour, lat, lon, elev)
        baseline: baseline forecast values [n]
        Returns calibrated forecast [n]
        """
        if var not in self.cache:
            self._load(var)
        model = self.cache[var]
        if model is None:
            # identity fallback
            return baseline
        # expected interface: model.predict(X) -> correction or calibrated value
        yhat = model.predict(X)
        # allow artifacts to be either absolute calibrated value or delta
        if yhat.shape == baseline.shape:
            return yhat
        if yhat.shape == (X.shape[0],):
            return yhat
        # if model outputs delta as single column
        if yhat.ndim == 2 and yhat.shape[1] == 1:
            return baseline + yhat[:, 0]
        return baseline
