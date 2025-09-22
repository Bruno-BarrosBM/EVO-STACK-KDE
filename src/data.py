from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplits:
    """Container for scaled data splits."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    scaler: StandardScaler

    def shapes(self) -> Tuple[int, int, int]:
        return self.X_train.shape[0], self.X_val.shape[0], self.X_test.shape[0]

    def dimension(self) -> int:
        return self.X_train.shape[1]


def load_wine_red(csv_path: str) -> pd.DataFrame:
    """Load the UCI Wine Quality (red) dataset with delimiter auto-detection."""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_file, sep=None, engine="python")
    except Exception:
        # Fall back to default comma delimiter if sniffing fails
        df = pd.read_csv(csv_file)

    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> DataSplits:
    """Split dataframe into train/val/test and apply standard scaling."""

    if "quality" in df.columns:
        df = df.drop(columns=["quality"])

    X = df.values.astype(float)
    X_train, X_temp = train_test_split(X, test_size=test_size + val_size, random_state=seed, shuffle=True)

    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test = train_test_split(X_temp, test_size=1 - relative_val_size, random_state=seed, shuffle=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return DataSplits(X_train_scaled, X_val_scaled, X_test_scaled, scaler)


def persist_splits(splits: DataSplits, out_dir: str) -> None:
    """Persist scaled splits and metadata to disk."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.savez(out_path / "splits.npz", X_train=splits.X_train, X_val=splits.X_val, X_test=splits.X_test)

    meta = {
        "n_train": int(splits.X_train.shape[0]),
        "n_val": int(splits.X_val.shape[0]),
        "n_test": int(splits.X_test.shape[0]),
        "dimension": int(splits.dimension()),
        "scaler_mean": splits.scaler.mean_.tolist(),
        "scaler_scale": splits.scaler.scale_.tolist(),
    }

    with open(out_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
