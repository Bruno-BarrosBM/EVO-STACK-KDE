from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_2d(pareto_fits: Iterable[Iterable[float]], out_png: Path) -> None:
    data = np.array(list(pareto_fits), dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("pareto_fits must be iterable of (f1, f2, f3)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(data[:, 0], data[:, 1], c="tab:blue", alpha=0.7)
    axes[0].set_xlabel("f1: NLL")
    axes[0].set_ylabel("f2: Stability")
    axes[0].set_title("Pareto Front (f1 vs f2)")

    axes[1].scatter(data[:, 0], data[:, 2], c="tab:orange", alpha=0.7)
    axes[1].set_xlabel("f1: NLL")
    axes[1].set_ylabel("f3: Complexity")
    axes[1].set_title("Pareto Front (f1 vs f3)")

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_hist_neglogp_test(neg_logp: Iterable[float], out_png: Path) -> None:
    values = np.array(list(neg_logp), dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=20, color="tab:green", alpha=0.8)
    ax.set_xlabel("-log p(x)")
    ax.set_ylabel("Frequency")
    ax.set_title("Test Set Negative Log-Likelihood")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
