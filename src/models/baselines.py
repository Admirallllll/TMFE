from __future__ import annotations

import numpy as np


def regression_no_change(y_true: np.ndarray) -> np.ndarray:
    return np.zeros_like(y_true, dtype=float)


def classification_majority(y_train: np.ndarray, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    p = float(np.mean(y_train))
    proba = np.full(shape=(n_test,), fill_value=p, dtype=float)
    pred = (proba >= 0.5).astype(int)
    return pred, proba

