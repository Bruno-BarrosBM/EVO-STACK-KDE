from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.model_selection import KFold

if False:  # type checking imports
    from .genome import ModelConfig


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("Expected 2D array")
    return X


def nll_kfold(model_factory: Callable[[], object], X: np.ndarray, k: int = 3, seed: int = 42) -> float:
    X = _ensure_2d(np.asarray(X, dtype=float))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    nlls = []
    for train_idx, val_idx in kf.split(X):
        model = model_factory()
        model.fit(X[train_idx])
        logp = model.logpdf(X[val_idx])
        nlls.append(float(-np.mean(logp)))
    return float(np.mean(nlls))


def stability_bootstrap(
    model_factory: Callable[[], object],
    X: np.ndarray,
    B: int = 10,
    k: int = 3,
    seed: int = 42,
) -> float:
    X = _ensure_2d(np.asarray(X, dtype=float))
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    scores = []
    for _ in range(B):
        sample_idx = rng.integers(0, n, size=n)
        X_boot = X[sample_idx]
        scores.append(nll_kfold(model_factory, X_boot, k=k, seed=seed))
    return float(np.std(scores))


def _softmax(z: Sequence[float]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def complexity(config: "ModelConfig", n: int, d: int, lam_entropy: float = 0.0) -> float:
    if d <= 0:
        raise ValueError("d must be positive")
    base_factor = min(1.0, n / 2000.0)
    complexities = []
    for expert in config.experts:
        mask = np.asarray(expert.feature_mask, dtype=bool)
        if not mask.any():
            raise ValueError("Expert mask must select at least one feature")
        complexities.append(mask.sum() / d * base_factor)
    mean_complexity = float(np.mean(complexities))
    if lam_entropy > 0:
        weights = _softmax(config.weight_logits)
        entropy = -np.sum(weights * np.log(weights + 1e-12))
        norm_entropy = entropy / np.log(len(weights))
        mean_complexity += lam_entropy * float(norm_entropy)
    return mean_complexity
