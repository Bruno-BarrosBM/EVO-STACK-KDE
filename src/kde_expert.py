from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import KernelDensity


def scott_bandwidth(X: np.ndarray) -> float:
    """Scott's rule of thumb bandwidth for Gaussian KDE."""

    if X.ndim != 2:
        raise ValueError("Input array must be 2D")
    n, d = X.shape
    if n <= 0 or d <= 0:
        raise ValueError("Input array must have positive dimensions")
    return np.power(n, -1.0 / (d + 4))


@dataclass
class KDEExpert:
    feature_mask: Optional[np.ndarray]
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.feature_mask is not None:
            mask = np.asarray(self.feature_mask, dtype=bool)
            if mask.ndim != 1:
                raise ValueError("feature_mask must be 1D")
            if not mask.any():
                raise ValueError("feature_mask must select at least one feature")
            self.feature_mask = mask
        self.model: Optional[KernelDensity] = None

    def _apply_mask(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mask is None:
            return X
        return X[:, self.feature_mask]

    def fit(self, X: np.ndarray) -> "KDEExpert":
        subspace = self._apply_mask(X)
        h0 = scott_bandwidth(subspace)
        bandwidth = max(h0 * float(self.alpha), 1e-6)
        self.model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self.model.fit(subspace)
        return self

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("KDEExpert must be fitted before calling logpdf")
        subspace = self._apply_mask(X)
        return self.model.score_samples(subspace)
