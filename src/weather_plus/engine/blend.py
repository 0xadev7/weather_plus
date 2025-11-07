from __future__ import annotations
import numpy as np


def simple_meta_blend(
    stacked_preds: np.ndarray, regime_weights: np.ndarray | None = None
) -> np.ndarray:
    """
    stacked_preds: [n, k] predictions from k level-1 learners (EMOS/QRF/etc)
    regime_weights: [k] or [n, k] weights; if None -> equal weights.
    Returns [n]
    """
    if stacked_preds.ndim != 2:
        return stacked_preds
    n, k = stacked_preds.shape
    if regime_weights is None:
        w = np.ones((n, k)) / k
    else:
        w = regime_weights
        if w.ndim == 1:
            w = np.tile(w[None, :], (n, 1))
        w = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
    return np.sum(stacked_preds * w, axis=1)
