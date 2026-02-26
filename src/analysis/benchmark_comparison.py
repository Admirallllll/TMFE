"""
Benchmark comparison for predicting AI initiation scores.

Produces slide-ready fold metrics, summary tables, and a comparison chart using
the same evaluation split across all models (default: GroupKFold by ticker).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, Lasso, LassoCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_METADATA_FEATURES = [
    "log_mktcap",
    "rd_intensity",
    "eps_positive",
    "stock_price",
    "year",
    "quarter",
    "sector",
]
DEFAULT_TEXT_RATIO_FEATURES = [
    "speech_kw_ai_ratio",
    "qa_kw_ai_ratio",
    "overall_kw_ai_ratio",
]


def _safe_kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Kendall's tau; fall back to 0 when undefined (e.g., constant prediction)."""
    try:
        tau, _ = kendalltau(y_true, y_pred)
    except Exception:
        return 0.0
    if tau is None or np.isnan(tau):
        return 0.0
    return float(tau)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(err ** 2)))


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _aggregate_doc_text(sentences_df: pd.DataFrame, section: Optional[str] = "qa") -> pd.DataFrame:
    if sentences_df is None or len(sentences_df) == 0:
        return pd.DataFrame(columns=["doc_id", "doc_text"])

    cols = ["doc_id", "text"] + (["section"] if "section" in sentences_df.columns else [])
    df = sentences_df[cols].copy()
    if "text" not in df.columns:
        return pd.DataFrame(columns=["doc_id", "doc_text"])

    if section and "section" in df.columns:
        filtered = df[df["section"] == section].copy()
        if len(filtered) > 0:
            df = filtered
    df["text"] = df["text"].fillna("").astype(str)
    grouped = df.groupby("doc_id", sort=False)["text"].agg(" ".join).reset_index()
    return grouped.rename(columns={"text": "doc_text"})


def _choose_metadata_columns(df: pd.DataFrame, requested: Optional[Sequence[str]]) -> List[str]:
    candidates = list(requested) if requested is not None else DEFAULT_METADATA_FEATURES
    return [c for c in candidates if c in df.columns]


def _build_metadata_pipeline(X_train_meta: pd.DataFrame, model_kind: str, random_state: int = 42) -> Pipeline:
    num_cols = [c for c in X_train_meta.columns if pd.api.types.is_numeric_dtype(X_train_meta[c])]
    cat_cols = [c for c in X_train_meta.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                cat_cols,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    if model_kind == "ols":
        model = LinearRegression()
    elif model_kind == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8, 1.0],
            cv=3,
            random_state=random_state,
            max_iter=5000,
        )
    else:
        raise ValueError(f"Unknown metadata model_kind: {model_kind}")

    return Pipeline([("pre", pre), ("model", model)])


def _predict_text_lasso(
    train_text: Sequence[str],
    y_train: np.ndarray,
    test_text: Sequence[str],
    text_max_features: int,
    random_state: int,
    inner_cv: int = 3,
    fixed_alpha: float = 1e-4,
) -> np.ndarray:
    train_text = pd.Series(train_text).fillna("").astype(str)
    test_text = pd.Series(test_text).fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=text_max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_text.tolist())
    X_test = vectorizer.transform(test_text.tolist())

    # Fallback if the corpus has no usable vocabulary (e.g. all stopwords).
    if X_train.shape[1] == 0:
        return np.full(len(test_text), float(np.mean(y_train)))

    # Large sparse LassoCV can be very slow in repeated fold evaluation.
    # Use a fixed-alpha Lasso for realistic datasets; keep tiny-sample CV tuning
    # for toy/test data where runtime is negligible.
    if len(y_train) <= 500:
        cv = min(inner_cv, len(y_train))
        if cv < 2:
            return np.full(len(test_text), float(np.mean(y_train)))
        model = LassoCV(cv=cv, random_state=random_state, max_iter=3000, n_jobs=1)
    else:
        model = Lasso(alpha=fixed_alpha, max_iter=3000)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _predict_text_ratio_lasso(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    text_ratio_features: Sequence[str],
) -> np.ndarray:
    cols = [c for c in text_ratio_features if c in train_df.columns]
    if not cols:
        return np.full(len(test_df), float(np.mean(y_train)))

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=1e-4, max_iter=2000)),
        ]
    )
    pipe.fit(train_df[cols], y_train)
    return pipe.predict(test_df[cols])


def _iter_splits(
    df: pd.DataFrame,
    group_col: Optional[str],
    n_splits: int,
    random_state: int,
) -> Tuple[Iterable[Tuple[np.ndarray, np.ndarray]], str]:
    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("__MISSING_GROUP__").astype(str)
        n_groups = groups.nunique()
        if n_groups >= n_splits and n_splits >= 2:
            splitter = GroupKFold(n_splits=n_splits)
            return splitter.split(df, groups=groups), f"GroupKFold({group_col})"

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return splitter.split(df), "KFold"


def evaluate_benchmark_models(
    regression_df: pd.DataFrame,
    sentences_df: Optional[pd.DataFrame] = None,
    target_col: str = "ai_initiation_score",
    group_col: str = "ticker",
    n_splits: int = 5,
    random_state: int = 42,
    text_section: Optional[str] = "qa",
    text_max_features: int = 3000,
    metadata_features: Optional[Sequence[str]] = None,
    include_metadata_elasticnet: bool = True,
    text_model_mode: str = "raw",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate baseline and benchmark models under shared CV splits.

    Returns:
        (fold_metrics_df, summary_df)
    """
    if target_col not in regression_df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if "doc_id" not in regression_df.columns:
        raise ValueError("regression_df must include 'doc_id'")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    base = regression_df.copy()
    base = base[base[target_col].notna()].copy()
    if text_model_mode == "raw":
        text_df = _aggregate_doc_text(sentences_df, section=text_section) if sentences_df is not None else pd.DataFrame(columns=["doc_id", "doc_text"])
        base = base.merge(text_df, on="doc_id", how="left")
        base["doc_text"] = base.get("doc_text", "").fillna("").astype(str)
    elif text_model_mode == "ratios":
        pass
    else:
        raise ValueError("text_model_mode must be 'raw' or 'ratios'")

    # Keep rows with valid target and doc_id. Group values may be missing and are handled in splitter.
    base = base[base["doc_id"].notna()].copy().reset_index(drop=True)
    if len(base) < n_splits:
        raise ValueError(f"Not enough rows ({len(base)}) for n_splits={n_splits}")

    meta_cols = _choose_metadata_columns(base, metadata_features)
    if len(meta_cols) == 0:
        raise ValueError("No metadata features available for benchmark evaluation")

    split_iter, split_method = _iter_splits(base, group_col=group_col, n_splits=n_splits, random_state=random_state)
    fold_records: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_df = base.iloc[train_idx].copy()
        test_df = base.iloc[test_idx].copy()
        if verbose:
            print(f"[Benchmark] Fold {fold_idx}/{n_splits}: train={len(train_df)} test={len(test_df)}")

        y_train = train_df[target_col].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)
        X_train_meta = train_df[meta_cols].copy()
        X_test_meta = test_df[meta_cols].copy()

        train_groups = set(train_df[group_col].fillna("__MISSING_GROUP__").astype(str)) if group_col in train_df.columns else set()
        test_groups = set(test_df[group_col].fillna("__MISSING_GROUP__").astype(str)) if group_col in test_df.columns else set()
        group_overlap_count = len(train_groups & test_groups) if group_col in base.columns else 0

        model_preds: Dict[str, np.ndarray] = {}

        mean_model = DummyRegressor(strategy="mean")
        mean_model.fit(np.zeros((len(train_df), 1)), y_train)
        model_preds["Mean Baseline"] = mean_model.predict(np.zeros((len(test_df), 1)))

        try:
            meta_ols = _build_metadata_pipeline(X_train_meta, model_kind="ols", random_state=random_state)
            meta_ols.fit(X_train_meta, y_train)
            model_preds["Metadata OLS"] = meta_ols.predict(X_test_meta)
        except Exception:
            model_preds["Metadata OLS"] = np.full(len(test_df), float(np.mean(y_train)))

        if include_metadata_elasticnet:
            try:
                if verbose:
                    print("  - fitting Metadata ElasticNet")
                meta_en = _build_metadata_pipeline(X_train_meta, model_kind="elasticnet", random_state=random_state)
                meta_en.fit(X_train_meta, y_train)
                model_preds["Metadata ElasticNet"] = meta_en.predict(X_test_meta)
            except Exception:
                model_preds["Metadata ElasticNet"] = np.full(len(test_df), float(np.mean(y_train)))

        try:
            if verbose:
                print(f"  - fitting Text Lasso ({text_model_mode})")
            if text_model_mode == "raw":
                model_preds["Text Lasso"] = _predict_text_lasso(
                    train_text=train_df["doc_text"],
                    y_train=y_train,
                    test_text=test_df["doc_text"],
                    text_max_features=text_max_features,
                    random_state=random_state,
                )
            else:
                model_preds["Text Lasso"] = _predict_text_ratio_lasso(
                    train_df=train_df,
                    y_train=y_train,
                    test_df=test_df,
                    text_ratio_features=DEFAULT_TEXT_RATIO_FEATURES,
                )
        except Exception:
            model_preds["Text Lasso"] = np.full(len(test_df), float(np.mean(y_train)))

        for model_name, y_pred in model_preds.items():
            y_pred = np.asarray(y_pred, dtype=float)
            fold_records.append(
                {
                    "Fold": fold_idx,
                    "Model": model_name,
                    "MAE": float(np.mean(np.abs(y_test - y_pred))),
                    "RMSE": _rmse(y_test, y_pred),
                    "Kendall Tau": _safe_kendall_tau(y_test, y_pred),
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "group_overlap_count": int(group_overlap_count),
                    "split_method": split_method,
                    "target_col": target_col,
                }
            )

    folds_df = pd.DataFrame(fold_records)
    if len(folds_df) == 0:
        raise RuntimeError("No fold results were produced")

    summary = (
        folds_df.groupby("Model", as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            **{
                "Kendall Tau_mean": ("Kendall Tau", "mean"),
                "Kendall Tau_std": ("Kendall Tau", "std"),
            },
            Folds=("Fold", "count"),
        )
        .sort_values(["MAE_mean", "Kendall Tau_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    for col in ["MAE_std", "RMSE_std", "Kendall Tau_std"]:
        summary[col] = summary[col].fillna(0.0)

    return folds_df, summary


def _plot_benchmark_comparison(summary_df: pd.DataFrame, output_png: str) -> None:
    plot_df = summary_df.copy()
    if len(plot_df) == 0:
        return

    # Preserve row order (best MAE first).
    y_pos = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(
        y_pos,
        plot_df["MAE_mean"].values,
        xerr=plot_df["MAE_std"].values,
        color="#4C78A8",
        alpha=0.9,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["Model"].tolist())
    axes[0].invert_yaxis()
    axes[0].set_title("MAE (lower is better)")
    axes[0].set_xlabel("MAE")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(
        y_pos,
        plot_df["Kendall Tau_mean"].values,
        xerr=plot_df["Kendall Tau_std"].values,
        color="#F58518",
        alpha=0.9,
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(plot_df["Model"].tolist())
    axes[1].invert_yaxis()
    axes[1].set_title("Kendall's Tau (higher is better)")
    axes[1].set_xlabel("Kendall Tau")
    axes[1].grid(axis="x", alpha=0.2)

    fig.suptitle("Benchmark Comparison: Predicting ai_initiation_score", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_benchmark_outputs(
    folds_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    prefix: str = "benchmark_comparison",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    folds_csv = os.path.join(output_dir, f"{prefix}_folds.csv")
    summary_csv = os.path.join(output_dir, f"{prefix}_summary.csv")
    plot_png = os.path.join(output_dir, f"{prefix}.png")

    folds_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _plot_benchmark_comparison(summary_df, plot_png)

    return {"folds_csv": folds_csv, "summary_csv": summary_csv, "plot_png": plot_png}


def run_benchmark_comparison(
    regression_dataset_path: str,
    sentences_path: str,
    output_dir: str = "outputs/figures",
    target_col: str = "ai_initiation_score",
    group_col: str = "ticker",
    n_splits: int = 5,
    random_state: int = 42,
    text_section: Optional[str] = "qa",
    text_max_features: int = 3000,
    include_metadata_elasticnet: bool = True,
    text_model_mode: str = "ratios",
    verbose: bool = True,
) -> Dict[str, str]:
    reg_df = pd.read_parquet(regression_dataset_path)
    sentences_df = pd.read_parquet(sentences_path)
    folds_df, summary_df = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=sentences_df,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        random_state=random_state,
        text_section=text_section,
        text_max_features=text_max_features,
        include_metadata_elasticnet=include_metadata_elasticnet,
        text_model_mode=text_model_mode,
        verbose=verbose,
    )
    paths = write_benchmark_outputs(folds_df, summary_df, output_dir=output_dir)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark comparison for ai_initiation_score prediction")
    parser.add_argument("--regression-dataset", default="outputs/features/regression_dataset.parquet")
    parser.add_argument("--sentences", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--target-col", default="ai_initiation_score")
    parser.add_argument("--group-col", default="ticker")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-section", default="qa", choices=["qa", "speech", "all"])
    parser.add_argument("--text-max-features", type=int, default=3000)
    parser.add_argument("--no-metadata-elasticnet", action="store_true")
    parser.add_argument("--text-model", default="ratios", choices=["ratios", "raw"])
    args = parser.parse_args()

    section = None if args.text_section == "all" else args.text_section
    paths = run_benchmark_comparison(
        regression_dataset_path=args.regression_dataset,
        sentences_path=args.sentences,
        output_dir=args.output_dir,
        target_col=args.target_col,
        group_col=args.group_col,
        n_splits=args.cv_folds,
        random_state=args.seed,
        text_section=section,
        text_max_features=args.text_max_features,
        include_metadata_elasticnet=not args.no_metadata_elasticnet,
        text_model_mode=args.text_model,
    )
    print("Saved benchmark outputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
