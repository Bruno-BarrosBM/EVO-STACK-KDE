from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np

from .ensemble import KDEEnsemble
from .kde_expert import KDEExpert


@dataclass
class ExpertConfig:
    alpha: float
    feature_mask: np.ndarray

    def to_jsonable(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "feature_mask": self.feature_mask.astype(int).tolist(),
        }


@dataclass
class ModelConfig:
    weight_logits: np.ndarray
    experts: List[ExpertConfig] = field(default_factory=list)

    def to_jsonable(self) -> dict:
        return {
            "weight_logits": self.weight_logits.tolist(),
            "experts": [expert.to_jsonable() for expert in self.experts],
        }


def random_mask(d: int, rng: np.random.Generator, keep_frac_range: Sequence[float] = (0.6, 0.8)) -> np.ndarray:
    if d <= 0:
        raise ValueError("d must be positive")
    low, high = keep_frac_range
    keep_frac = rng.uniform(low, high)
    n_keep = max(1, int(round(keep_frac * d)))
    mask = np.zeros(d, dtype=bool)
    idx = rng.choice(d, size=n_keep, replace=False)
    mask[idx] = True
    return mask


def random_model_config(d: int, rng: np.random.Generator, m: int = 5) -> ModelConfig:
    experts = []
    for _ in range(m):
        alpha = float(rng.uniform(0.5, 1.5))
        mask = random_mask(d, rng)
        experts.append(ExpertConfig(alpha=alpha, feature_mask=mask))
    weight_logits = rng.normal(0.0, 1.0, size=m)
    return ModelConfig(weight_logits=weight_logits, experts=experts)


def decode_to_model(config: ModelConfig) -> Callable[[], KDEEnsemble]:
    def factory() -> KDEEnsemble:
        experts = [KDEExpert(feature_mask=cfg.feature_mask.copy(), alpha=cfg.alpha) for cfg in config.experts]
        return KDEEnsemble(experts=experts, weight_logits=np.array(config.weight_logits, dtype=float))

    return factory


def model_config_from_jsonable(data: dict) -> ModelConfig:
    experts = [
        ExpertConfig(alpha=float(exp["alpha"]), feature_mask=np.array(exp["feature_mask"], dtype=bool))
        for exp in data["experts"]
    ]
    weight_logits = np.array(data["weight_logits"], dtype=float)
    return ModelConfig(weight_logits=weight_logits, experts=experts)
