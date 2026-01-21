from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.models.baselines import classification_majority, regression_no_change
from src.models.train_classification import build_classification_models
from src.models.train_regression import build_regression_models
from src.utils.time import datacqtr_to_index


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    train_end: str
    test_start: str


def time_split(
    df: pd.DataFrame,
    *,
    train_end_datacqtr: str,
    test_start_datacqtr: str,
) -> SplitData:
    train_end_idx = datacqtr_to_index(train_end_datacqtr)
    test_start_idx = datacqtr_to_index(test_start_datacqtr)

    train = df.loc[df["quarter_index"].astype(int) <= train_end_idx].sort_values(["quarter_index", "ticker"]).reset_index(drop=True)
    test = df.loc[df["quarter_index"].astype(int) >= test_start_idx].sort_values(["quarter_index", "ticker"]).reset_index(drop=True)
    return SplitData(train=train, test=test, train_end=train_end_datacqtr, test_start=test_start_datacqtr)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_pred = (y_proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_true, y_proba))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    return {"auc": auc, "f1": f1, "accuracy": acc}


def _rolling_splits(
    quarter_indices: list[int],
    *,
    min_train_quarters: int,
    test_quarters: int,
    step_quarters: int,
) -> list[tuple[int, int]]:
    splits = []
    if len(quarter_indices) < (min_train_quarters + test_quarters):
        return splits
    start = min_train_quarters - 1
    while (start + test_quarters) < len(quarter_indices):
        train_end = quarter_indices[start]
        test_end = quarter_indices[start + test_quarters]
        splits.append((train_end, test_end))
        start += step_quarters
    return splits


def rolling_evaluation_indices(
    df: pd.DataFrame,
    *,
    min_train_quarters: int = 20,
    test_quarters: int = 4,
    step_quarters: int = 4,
) -> list[tuple[pd.Index, pd.Index, str, str]]:
    q = sorted(df["quarter_index"].dropna().astype(int).unique().tolist())
    splits = _rolling_splits(q, min_train_quarters=min_train_quarters, test_quarters=test_quarters, step_quarters=step_quarters)
    out = []
    for train_end_idx, test_end_idx in splits:
        train_mask = df["quarter_index"].astype(int) <= train_end_idx
        test_mask = (df["quarter_index"].astype(int) > train_end_idx) & (df["quarter_index"].astype(int) <= test_end_idx)
        train_q = df.loc[train_mask, "datacqtr"].astype(str).max()
        test_q = df.loc[test_mask, "datacqtr"].astype(str).max()
        out.append((df.index[train_mask], df.index[test_mask], str(train_q), str(test_q)))
    return out


def run_model_benchmarks(
    df: pd.DataFrame,
    *,
    target_reg: str,
    target_cls: str,
    numeric_features_meta: list[str],
    numeric_features_text: list[str],
    categorical_features: list[str],
    train_end_datacqtr: str,
    test_start_datacqtr: str,
    seed: int,
    model_dir: Path,
    logger,
) -> pd.DataFrame:
    split = time_split(df, train_end_datacqtr=train_end_datacqtr, test_start_datacqtr=test_start_datacqtr)

    y_train_reg = split.train[target_reg].astype(float).to_numpy()
    y_test_reg = split.test[target_reg].astype(float).to_numpy()
    y_train_cls = split.train[target_cls].astype(int).to_numpy()
    y_test_cls = split.test[target_cls].astype(int).to_numpy()

    q_low, q_high = 0.01, 0.99
    winsor_lo, winsor_hi = np.quantile(y_train_reg, [q_low, q_high])
    y_train_reg_fit = np.clip(y_train_reg, winsor_lo, winsor_hi)
    y_test_reg_eval = np.clip(y_test_reg, winsor_lo, winsor_hi)
    logger.info(f"Regression target winsorization (train-only): q{int(q_low*100)}={winsor_lo:.3f}, q{int(q_high*100)}={winsor_hi:.3f}")

    rows: list[dict[str, object]] = []

    bm1_pred = regression_no_change(y_test_reg_eval)
    rows.append(
        {
            "task": "regression",
            "model": "BM1_no_change",
            **regression_metrics(y_test_reg_eval, bm1_pred),
        }
    )

    maj_pred, maj_proba = classification_majority(y_train_cls, len(y_test_cls))
    rows.append(
        {
            "task": "classification",
            "model": "BM1_majority",
            **classification_metrics(y_test_cls, maj_proba),
            "pos_rate_train": float(np.mean(y_train_cls)),
            "pos_rate_test": float(np.mean(y_test_cls)),
        }
    )

    dense_cats = [c for c in categorical_features if c != "ticker"]

    bm2_reg_models = build_regression_models(
        numeric_features=numeric_features_meta,
        categorical_features=categorical_features,
        seed=seed,
    )[:1]
    m_reg_models = build_regression_models(
        numeric_features=numeric_features_text,
        categorical_features=categorical_features,
        categorical_features_dense=dense_cats,
        seed=seed,
    )[1:]

    reg_models = bm2_reg_models + m_reg_models
    for spec in reg_models:
        logger.info(f"Fitting {spec.key} (regression)")
        spec.pipeline.fit(split.train, y_train_reg_fit)
        y_pred = spec.pipeline.predict(split.test)
        dump(spec.pipeline, model_dir / f"{spec.key}__regression.joblib")
        rows.append({"task": "regression", "model": spec.key, **regression_metrics(y_test_reg_eval, y_pred)})

    bm2_cls_models = build_classification_models(
        numeric_features=numeric_features_meta,
        categorical_features=categorical_features,
        seed=seed,
    )[:1]
    m_cls_models = build_classification_models(
        numeric_features=numeric_features_text,
        categorical_features=categorical_features,
        categorical_features_dense=dense_cats,
        seed=seed,
    )[1:]

    cls_models = bm2_cls_models + m_cls_models
    for spec in cls_models:
        logger.info(f"Fitting {spec.key} (classification)")
        spec.pipeline.fit(split.train, y_train_cls)
        if hasattr(spec.pipeline, "predict_proba"):
            proba = spec.pipeline.predict_proba(split.test)[:, 1]
        else:
            proba = spec.pipeline.predict(split.test)
        dump(spec.pipeline, model_dir / f"{spec.key}__classification.joblib")
        rows.append({"task": "classification", "model": spec.key, **classification_metrics(y_test_cls, proba)})

    return pd.DataFrame(rows)


def run_rolling_benchmarks(
    df: pd.DataFrame,
    *,
    target_reg: str,
    target_cls: str,
    numeric_features_meta: list[str],
    numeric_features_text: list[str],
    categorical_features: list[str],
    seed: int,
    logger,
) -> pd.DataFrame:
    folds = rolling_evaluation_indices(df)
    if not folds:
        logger.info("Rolling evaluation skipped (not enough quarters)")
        return pd.DataFrame()

    reg_specs = [
        ("BM2_metadata_ridge", build_regression_models(numeric_features=numeric_features_meta, categorical_features=categorical_features, seed=seed)[0]),
        ("M1_text_elasticnet", build_regression_models(numeric_features=numeric_features_text, categorical_features=categorical_features, seed=seed)[1]),
    ]
    cls_specs = [
        ("BM2_metadata_logit", build_classification_models(numeric_features=numeric_features_meta, categorical_features=categorical_features, seed=seed)[0]),
        ("M1_text_logitl1", build_classification_models(numeric_features=numeric_features_text, categorical_features=categorical_features, seed=seed)[1]),
    ]

    rows: list[dict[str, object]] = []

    for i, (train_idx, test_idx, train_q, test_q) in enumerate(folds, start=1):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        y_train_reg = train[target_reg].astype(float).to_numpy()
        y_test_reg = test[target_reg].astype(float).to_numpy()
        y_train_cls = train[target_cls].astype(int).to_numpy()
        y_test_cls = test[target_cls].astype(int).to_numpy()

        winsor_lo, winsor_hi = np.quantile(y_train_reg, [0.01, 0.99])
        y_train_reg_fit = np.clip(y_train_reg, winsor_lo, winsor_hi)
        y_test_reg_eval = np.clip(y_test_reg, winsor_lo, winsor_hi)

        for key, spec in reg_specs:
            spec.pipeline.fit(train, y_train_reg_fit)
            y_pred = spec.pipeline.predict(test)
            m = regression_metrics(y_test_reg_eval, y_pred)
            rows.append({"fold": i, "task": "regression", "model": key, "train_end": train_q, "test_end": test_q, **m})

        for key, spec in cls_specs:
            spec.pipeline.fit(train, y_train_cls)
            proba = spec.pipeline.predict_proba(test)[:, 1]
            m = classification_metrics(y_test_cls, proba)
            rows.append({"fold": i, "task": "classification", "model": key, "train_end": train_q, "test_end": test_q, **m})

    return pd.DataFrame(rows)
