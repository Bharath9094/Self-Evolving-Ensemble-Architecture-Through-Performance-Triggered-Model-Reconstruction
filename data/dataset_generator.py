"""Reproducible concept-drift classification data generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
def generate_dataset(
    name: str,
    n_samples: int = 1500,
    n_features: int = 6,
    noise: float = 0.05,
    drift_location: float = 0.5,
    drift_severity: float = 1.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Generate an ordered binary stream with an explicit target column."""
    if n_samples < 30:
        raise ValueError("At least 30 samples are required.")
    if n_features < 2:
        raise ValueError("At least 2 features are required.")
    if not 0.05 <= drift_location <= 0.95:
        raise ValueError("Drift location must be between 0.05 and 0.95.")
    if not 0 <= noise <= 0.5:
        raise ValueError("Noise must be between 0 and 0.5.")
    if drift_severity < 0:
        raise ValueError("Drift severity cannot be negative.")

    rng = np.random.default_rng(random_seed)
    X = rng.uniform(0, 10, size=(n_samples, n_features))
    cut = int(n_samples * drift_location)
    mode = name.lower()

    if "hyperplane" in mode:
        X = rng.uniform(-1, 1, size=(n_samples, n_features))
        weights = rng.normal(size=n_features)
        y = np.zeros(n_samples, dtype=int)
        for i in range(cut, n_samples):
            progress = (i - cut) / max(1, n_samples - cut)
            rotated = np.roll(weights, 1)
            change = min(1.0, progress * drift_severity)
            current = (1 - change) * weights + change * rotated
            y[i] = int(X[i] @ current >= 0)
        y[:cut] = (X[:cut] @ weights >= 0).astype(int)
    elif "recurring" in mode or "sea" in mode:
        base_threshold = 8.0
        shifted_threshold = base_threshold + 4.0 * min(drift_severity, 2.0)
        y = (X[:, 0] + X[:, 1] <= base_threshold).astype(int)
        if "recurring" in mode:
            return_from = min(n_samples, cut + max(1, (n_samples - cut) // 2))
            y[cut:return_from] = (X[cut:return_from, 0] + X[cut:return_from, 1] <= shifted_threshold).astype(int)
        elif "gradual" in mode:
            transition = max(1, int(n_samples * 0.15))
            stop = min(n_samples, cut + transition)
            for i in range(cut, stop):
                fraction = (i - cut + 1) / transition
                threshold = base_threshold + fraction * (shifted_threshold - base_threshold)
                y[i] = int(X[i, 0] + X[i, 1] <= threshold)
            y[stop:] = (X[stop:, 0] + X[stop:, 1] <= shifted_threshold).astype(int)
        else:
            y[cut:] = (X[cut:, 0] + X[cut:, 1] <= shifted_threshold).astype(int)
        flips = rng.random(n_samples) < noise
        y[flips] = 1 - y[flips]
    else:
        weights = rng.normal(size=n_features)
        y = (X @ weights >= 0).astype(int)
        y[cut:] = (X[cut:] @ np.roll(weights, 1) >= 0).astype(int)
        flips = rng.random(n_samples) < noise
        y[flips] = 1 - y[flips]
    columns = [f"feature_{i + 1}" for i in range(n_features)]
    frame = pd.DataFrame(X, columns=columns)
    frame["target"] = y.astype(int)
    return frame, "target"


def generate_sea_dataset(
    n_samples: int = 1500,
    n_features: int = 6,
    noise: float = 0.05,
    drift_location: float = 0.5,
    drift_severity: float = 1.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    return generate_dataset(
        "SEA Concept Drift Dataset", n_samples, n_features, noise,
        drift_location, drift_severity, random_seed,
    )

