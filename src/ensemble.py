from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from .kde_expert import KDEExpert


@dataclass
class KDEEnsemble:
    experts: Sequence[KDEExpert]
    weight_logits: np.ndarray

    def __post_init__(self) -> None:
        if len(self.experts) == 0:
            raise ValueError("KDEEnsemble requires at least one expert")
        self.weight_logits = np.asarray(self.weight_logits, dtype=float)
        if self.weight_logits.shape != (len(self.experts),):
            raise ValueError("weight_logits must match number of experts")
        self._weights = self._softmax(self.weight_logits)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def fit(self, X: np.ndarray) -> "KDEEnsemble":
        for expert in self.experts:
            expert.fit(X)
        return self

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        logps = np.column_stack([expert.logpdf(X) for expert in self.experts])
        log_weights = np.log(self._weights)
        return logsumexp(logps + log_weights, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return density estimates for the provided samples."""

        log_density = self.logpdf(X)
        return np.exp(log_density)
