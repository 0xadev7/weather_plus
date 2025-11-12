import os, joblib
from fastapi import HTTPException


class ModelCache:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.memo = {}

    def _paths(self, var: str, tile: str):
        return [
            os.path.join(self.model_dir, f"{var}__{tile}.joblib"),
            os.path.join(self.model_dir, f"{var}.joblib"),
        ]

    def load(self, var: str, tile: str):
        key = (var, tile)
        if key in self.memo:
            return self.memo[key]
        for p in self._paths(var, tile):
            if os.path.exists(p):
                obj = joblib.load(p)
                self.memo[key] = obj
                return obj
        raise HTTPException(
            404, f"No model for {var} (tile {tile}) or global fallback."
        )

    def clear(self):
        self.memo.clear()
