from __future__ import annotations
import os
from typing import Optional, Dict, Any
import numpy as np
from joblib import load


class Calibrator:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.cache: Dict[str, Any] = {}

    def _paths_for(self, var: str, tile: Optional[str]):
        paths = []
        if tile:
            paths.append(os.path.join(self.model_dir, f"{var}__{tile}.joblib"))
        paths.append(os.path.join(self.model_dir, f"{var}.joblib"))  # global fallback
        return paths

    def _load_best(self, var: str, tile: Optional[str]):
        # cache key: (var, tile or "global")
        key = (var, tile or "global")
        if key in self.cache:
            return self.cache[key]
        for p in self._paths_for(var, tile):
            if os.path.exists(p):
                self.cache[key] = load(p)
                return self.cache[key]
        self.cache[key] = None
        return None

    def predict(
        self, var: str, X: np.ndarray, baseline: np.ndarray, tile: Optional[str] = None
    ) -> np.ndarray:
        m = self._load_best(var, tile)
        if m is None:
            return baseline  # identity
        yhat = m.predict(X)
        # allow delta or absolute
        if yhat.shape == baseline.shape:
            return yhat
        if yhat.ndim == 2 and yhat.shape[1] == 1:
            return baseline + yhat[:, 0]
        return yhat
