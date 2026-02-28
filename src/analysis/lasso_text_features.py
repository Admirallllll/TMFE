"""
Lasso Text Features Module

Uses N-gram (TF-IDF) features from earnings call text to predict AI-related
scores via Lasso regression, then produces:
  1. Coefficient plot (signed bar chart)
  2. Volcano Plot (x = coefficient, y = log document frequency) — the
     professor's signature visualization for identifying key linguistic drivers.

Usage (standalone):
    python -m src.analysis.lasso_text_features \
        --sentences outputs/features/sentences_with_keywords.parquet \
        --metrics  outputs/features/document_metrics.parquet \
        --output-dir outputs/figures

Usage (as a library):
    from src.analysis.lasso_text_features import run_lasso_text_analysis
    results = run_lasso_text_analysis(sentences_path, doc_metrics_path)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau

try:
    from src.utils.visual_style import (
        SPOTIFY_COLORS,
        apply_spotify_theme,
        save_figure,
        style_axes,
        style_legend,
    )
except Exception:  # pragma: no cover
    SPOTIFY_COLORS = {
        "background": "#121212",
        "fg": "#F5F5F5",
        "muted": "#B3B3B3",
        "accent": "#1DB954",
        "negative": "#FF5A5F",
        "grid": "#2A2A2A",
        "blue": "#4EA1FF",
    }
    def apply_spotify_theme():
        return None
    def style_axes(ax, **kwargs):
        return ax
    def style_legend(ax):
        return ax.get_legend()
    def save_figure(fig, output_path: str, dpi: int = 150):
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_TFIDF_MIN_DF = 0.01
DEFAULT_TFIDF_MAX_DF = 0.80
DEFAULT_LASSO_INNER_CV = 3
DEFAULT_LASSO_N_ALPHAS = 50
DEFAULT_LASSO_MAX_ITER = 1000
DEFAULT_LASSO_TOL = 1e-3
DEFAULT_MODEL_N_JOBS = -1


def _build_lasso_text_pipeline(
    inner_cv: int,
    max_features: int,
    ngram_range: Tuple[int, int],
    random_state: int,
    tfidf_min_df: float = DEFAULT_TFIDF_MIN_DF,
    tfidf_max_df: float = DEFAULT_TFIDF_MAX_DF,
    lasso_n_alphas: int = DEFAULT_LASSO_N_ALPHAS,
    lasso_max_iter: int = DEFAULT_LASSO_MAX_ITER,
    lasso_tol: float = DEFAULT_LASSO_TOL,
    lasso_n_jobs: int = DEFAULT_MODEL_N_JOBS,
) -> Pipeline:
    """Build a sparse-safe TF-IDF -> scaler -> LassoCV pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                    min_df=tfidf_min_df,
                    max_df=tfidf_max_df,
                ),
            ),
            # Preserve sparsity; centering would densify the TF-IDF matrix and explode RAM.
            ("scaler", StandardScaler(with_mean=False)),
            (
                "lasso",
                LassoCV(
                    cv=inner_cv,
                    random_state=random_state,
                    n_alphas=lasso_n_alphas,
                    max_iter=lasso_max_iter,
                    tol=lasso_tol,
                    n_jobs=lasso_n_jobs,
                    verbose=0,
                ),
            ),
        ]
    )


def _parse_doc_id(doc_id: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Extract ticker, year, quarter from doc_id like 'AAPL_2023Q1'."""
    parts = str(doc_id).rsplit("_", 1)
    if len(parts) != 2:
        return None, None, None
    ticker = parts[0]
    yq = parts[1]
    if "Q" not in yq:
        return ticker, None, None
    try:
        year = int(yq.split("Q")[0])
        quarter = int(yq.split("Q")[1])
        return ticker, year, quarter
    except Exception:
        return ticker, None, None


def _build_doc_corpus(
    sentences_df: pd.DataFrame,
    section: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate sentence-level text to document level.

    Args:
        sentences_df: Sentence-level DataFrame with columns ['doc_id', 'text', 'section'].
        section: If provided, filter to this section ('speech' or 'qa').

    Returns:
        DataFrame with ['doc_id', 'text'] where text is the concatenated document.
    """
    df = sentences_df[["doc_id", "text"] + (["section"] if "section" in sentences_df.columns else [])].copy()
    if section and "section" in df.columns:
        df = df[df["section"] == section]

    df["text"] = df["text"].fillna("").astype(str)
    corpus = (
        df.groupby("doc_id", sort=False)["text"]
        .agg(" ".join)
        .reset_index()
    )
    return corpus


def _doc_frequencies(vectorizer: TfidfVectorizer, X_bin: np.ndarray) -> np.ndarray:
    """Return document-frequency counts for each vocabulary term."""
    return np.asarray(X_bin.sum(axis=0)).flatten()


def _precompute_corpus_features(
    corpus_df: pd.DataFrame,
    text_col: str = "text",
    doc_id_col: str = "doc_id",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: float = DEFAULT_TFIDF_MIN_DF,
    max_df: float = DEFAULT_TFIDF_MAX_DF,
) -> Dict:
    """
    Fit TF-IDF and scaling once for a corpus, then reuse across multiple targets.
    """
    base = corpus_df[[doc_id_col, text_col]].copy().reset_index(drop=True)
    base[text_col] = base[text_col].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
    )
    X = vectorizer.fit_transform(base[text_col].tolist())

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)

    doc_lookup = pd.DataFrame(
        {
            doc_id_col: base[doc_id_col].values,
            "_row_idx": np.arange(len(base), dtype=np.int32),
        }
    )

    return {
        "doc_lookup": doc_lookup,
        "X_scaled": X_scaled,
        "vectorizer": vectorizer,
        "doc_id_col": doc_id_col,
        "text_col": text_col,
    }


# ---------------------------------------------------------------------------
# Core modelling
# ---------------------------------------------------------------------------

def fit_lasso_ngram(
    corpus_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
    text_col: str = "text",
    doc_id_col: str = "doc_id",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    cv: int = 5,
    random_state: int = 42,
    precomputed_features: Optional[Dict] = None,
    compute_cv_predictions: bool = True,
    tfidf_min_df: float = DEFAULT_TFIDF_MIN_DF,
    tfidf_max_df: float = DEFAULT_TFIDF_MAX_DF,
    lasso_inner_cv: int = DEFAULT_LASSO_INNER_CV,
    lasso_n_alphas: int = DEFAULT_LASSO_N_ALPHAS,
    lasso_max_iter: int = DEFAULT_LASSO_MAX_ITER,
    lasso_tol: float = DEFAULT_LASSO_TOL,
    lasso_n_jobs: int = DEFAULT_MODEL_N_JOBS,
    oof_n_jobs: int = DEFAULT_MODEL_N_JOBS,
) -> Dict:
    """
    Fit a TF-IDF → LassoCV pipeline predicting *target_col* from text.

    Args:
        corpus_df:    DataFrame with [doc_id, text].
        target_df:    DataFrame with [doc_id, target_col].
        target_col:   Name of the outcome variable (e.g. 'overall_kw_ai_ratio').
        max_features: Maximum vocabulary size for TF-IDF.
        ngram_range:  N-gram range for TF-IDF vectorizer.
        cv:           Number of cross-validation folds.

    Returns:
        dict with keys: 'coef_df', 'vectorizer', 'lasso', 'y_true', 'y_pred',
                        'alpha', 'r2', 'kendall_tau', 'kendall_p'
    """
    # Keep API compatibility: precomputed_features may still be supplied by callers,
    # but OOF evaluation intentionally uses raw text + fold-local preprocessing to
    # avoid leakage. Final coefficient extraction is also fit on the merged corpus.
    merged = corpus_df[[doc_id_col, text_col]].merge(
        target_df[[doc_id_col, target_col]].dropna(), on=doc_id_col, how="inner"
    )
    if len(merged) < 30:
        print(f"[Lasso] Insufficient data for '{target_col}': only {len(merged)} documents.")
        return {}

    merged[text_col] = merged[text_col].fillna("").astype(str)
    texts = merged[text_col].tolist()
    y = merged[target_col].to_numpy(dtype=float)

    # Cross-validated out-of-fold predictions for unbiased Kendall Tau
    y_pred = None
    tau = None
    p_val = None
    r2_oof = None
    if compute_cv_predictions:
        n_outer = min(max(2, int(cv)), len(y))
        if n_outer >= 2:
            print(
                "  Calculating leakage-free OOF predictions for Kendall Tau "
                "(nested TF-IDF/scaler/LassoCV, parallel outer CV)..."
            )
            splitter = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
            splits = list(splitter.split(texts))
            min_train_size = min((len(train_idx) for train_idx, _ in splits), default=0)
            inner_cv_oof = min(max(2, int(lasso_inner_cv)), min_train_size)

            if inner_cv_oof < 2:
                y_pred = np.full(len(y), float(np.mean(y)), dtype=float)
            else:
                text_array = np.asarray(texts, dtype=object)
                try:
                    oof_estimator = _build_lasso_text_pipeline(
                        inner_cv=inner_cv_oof,
                        max_features=max_features,
                        ngram_range=ngram_range,
                        random_state=random_state,
                        tfidf_min_df=tfidf_min_df,
                        tfidf_max_df=tfidf_max_df,
                        lasso_n_alphas=lasso_n_alphas,
                        lasso_max_iter=lasso_max_iter,
                        lasso_tol=lasso_tol,
                        lasso_n_jobs=lasso_n_jobs,
                    )
                    y_pred = cross_val_predict(
                        oof_estimator,
                        text_array,
                        y,
                        cv=splits,
                        method="predict",
                        n_jobs=oof_n_jobs,
                    )
                except Exception:
                    # Fallback keeps OOF evaluation robust if a fold has an empty vocabulary.
                    y_pred = np.full(len(y), np.nan, dtype=float)
                    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
                        train_text = [texts[i] for i in train_idx]
                        test_text = [texts[i] for i in test_idx]
                        y_train = y[train_idx]
                        inner_cv_fold = min(max(2, int(lasso_inner_cv)), len(train_idx))
                        if inner_cv_fold < 2:
                            y_pred[test_idx] = float(np.mean(y_train))
                            continue
                        try:
                            pipe = _build_lasso_text_pipeline(
                                inner_cv=inner_cv_fold,
                                max_features=max_features,
                                ngram_range=ngram_range,
                                random_state=random_state,
                                tfidf_min_df=tfidf_min_df,
                                tfidf_max_df=tfidf_max_df,
                                lasso_n_alphas=lasso_n_alphas,
                                lasso_max_iter=lasso_max_iter,
                                lasso_tol=lasso_tol,
                                lasso_n_jobs=lasso_n_jobs,
                            )
                            pipe.fit(train_text, y_train)
                            y_pred[test_idx] = pipe.predict(test_text)
                        except Exception:
                            y_pred[test_idx] = float(np.mean(y_train))
                        if fold_idx == 1 and len(test_idx) > 0:
                            # Keep a lightweight progress trace for long runs.
                            print(f"    Fold {fold_idx}/{n_outer}: train={len(train_idx)} test={len(test_idx)}")
            valid = ~np.isnan(y_pred)
            if valid.sum() >= 2:
                tau, p_val = kendalltau(y[valid], y_pred[valid])
                ss_res = float(np.sum((y[valid] - y_pred[valid]) ** 2))
                ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
                r2_oof = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Final full-sample fit for coefficient interpretation (not for OOF evaluation).
    try:
        final_inner_cv = min(max(2, int(lasso_inner_cv)), len(y))
        final_pipe = _build_lasso_text_pipeline(
            inner_cv=final_inner_cv,
            max_features=max_features,
            ngram_range=ngram_range,
            random_state=random_state,
            tfidf_min_df=tfidf_min_df,
            tfidf_max_df=tfidf_max_df,
            lasso_n_alphas=lasso_n_alphas,
            lasso_max_iter=lasso_max_iter,
            lasso_tol=lasso_tol,
            lasso_n_jobs=lasso_n_jobs,
        )
        final_pipe.fit(texts, y)
        vectorizer = final_pipe.named_steps["tfidf"]
        scaler = final_pipe.named_steps["scaler"]
        lasso = final_pipe.named_steps["lasso"]
        X = vectorizer.transform(texts)
        X_scaled = scaler.transform(X)
        X_bin = (X > 0).astype(np.int8)
        doc_freq = _doc_frequencies(vectorizer, X_bin)
        r2 = float(final_pipe.score(texts, y))
        alpha = float(getattr(lasso, "alpha_", getattr(lasso, "alpha", np.nan)))
        print(f"  Training final LassoCV pipeline on {X_scaled.shape[0]} samples x {X_scaled.shape[1]} features...")
    except ValueError as e:
        print(f"[Lasso] Could not fit final pipeline for '{target_col}': {e}")
        return {}

    # Build coefficient DataFrame
    feature_names = vectorizer.get_feature_names_out()
    coefs = lasso.coef_

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "doc_frequency": doc_freq,
        "log_doc_frequency": np.log1p(doc_freq),
    })
    coef_df = coef_df[coef_df["coefficient"] != 0].copy()
    coef_df = coef_df.sort_values("coefficient", ascending=False)

    print(f"\n[Lasso → {target_col}]")
    print(f"  Alpha (best):   {alpha:.6f}")
    print(f"  R² (train):     {r2:.4f}")
    if r2_oof is not None:
        print(f"  R² (OOF):       {r2_oof:.4f}")
    tau_str = f"{tau:.4f}" if tau is not None else "skipped"
    p_str = f"{p_val:.4f}" if p_val is not None else "skipped"
    print(f"  Kendall's Tau (OOF):  {tau_str}  (p={p_str})")
    print(f"  Non-zero coefs: {(coefs != 0).sum()} / {len(coefs)}")

    if len(coef_df) > 0:
        print("\n  Top 10 POSITIVE features (management language → higher score):")
        print(coef_df.head(10)[["feature", "coefficient", "doc_frequency"]].to_string(index=False))
        print("\n  Top 10 NEGATIVE features (analyst language → lower score):")
        print(coef_df.tail(10)[["feature", "coefficient", "doc_frequency"]].to_string(index=False))

    return {
        "coef_df": coef_df,
        "vectorizer": vectorizer,
        "lasso": lasso,
        "y_true": y,
        "y_pred": y_pred,
        "alpha": alpha,
        "r2": r2,
        "r2_train": r2,
        "r2_oof": r2_oof,
        "kendall_tau": tau,
        "kendall_tau_oof": tau,
        "kendall_p": p_val,
        "target_col": target_col,
        "n_docs": len(merged),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_volcano(
    coef_df: pd.DataFrame,
    output_path: str,
    target_col: str = "target",
    top_n_labels: int = 15,
) -> None:
    """
    Volcano Plot: x = Lasso coefficient, y = log document frequency.
    The professor's signature visualization for linguistic key-driver analysis.

    Colour coding:
      - Green  : positive coefficient (associated with HIGHER score)
      - Red    : negative coefficient (associated with LOWER score)
      - Points above the median log-freq on either side get labelled.
    """
    if coef_df is None or len(coef_df) == 0:
        print("No non-zero coefficients to plot.")
        return
    apply_spotify_theme()

    df = coef_df.copy()
    df["color"] = df["coefficient"].apply(
        lambda c: SPOTIFY_COLORS.get("accent", "#1DB954") if c > 0 else SPOTIFY_COLORS.get("negative", "#FF5A5F")
    )
    df["abs_coef"] = df["coefficient"].abs()

    # Label the most impactful words on each side
    pos = df[df["coefficient"] > 0].nlargest(top_n_labels, "abs_coef")
    neg = df[df["coefficient"] < 0].nlargest(top_n_labels, "abs_coef")
    to_label = pd.concat([pos, neg])

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    ax.scatter(
        df["coefficient"],
        df["log_doc_frequency"],
        c=df["color"],
        alpha=0.65,
        s=30,
        linewidths=0,
    )

    # Annotate top features
    for _, row in to_label.iterrows():
        ax.annotate(
            row["feature"],
            xy=(row["coefficient"], row["log_doc_frequency"]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
            color=row["color"],
            alpha=0.9,
        )

    ax.axvline(0, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linewidth=0.8, linestyle="--")

    ax.set_xlabel("Lasso Coefficient\n← Lower AI Initiation         Higher AI Initiation →", fontsize=11)
    ax.set_ylabel("log(Document Frequency + 1)", fontsize=11)
    ax.set_title(
        f"Volcano Plot — N-gram Lasso Features\nTarget: {target_col}",
        fontsize=13,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SPOTIFY_COLORS.get("accent", "#1DB954"),
               markersize=8, label="Increases AI Initiation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SPOTIFY_COLORS.get("negative", "#FF5A5F"),
               markersize=8, label="Decreases AI Initiation"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    style_axes(ax, grid_axis="y", grid_alpha=0.10)
    style_legend(ax)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved Volcano Plot → {output_path}")


def plot_top_coefficients(
    coef_df: pd.DataFrame,
    output_path: str,
    target_col: str = "target",
    top_n: int = 20,
) -> None:
    """
    Horizontal bar chart of the top-N positive and top-N negative Lasso coefficients.
    Complementary to the Volcano Plot for easy readability.
    """
    if coef_df is None or len(coef_df) == 0:
        print("No coefficients to plot.")
        return
    apply_spotify_theme()

    pos = coef_df.nlargest(top_n, "coefficient")
    neg = coef_df.nsmallest(top_n, "coefficient")
    combined = pd.concat([neg, pos]).drop_duplicates("feature")
    combined = combined.sort_values("coefficient")

    colors = [
        SPOTIFY_COLORS.get("negative", "#FF5A5F") if c < 0 else SPOTIFY_COLORS.get("accent", "#1DB954")
        for c in combined["coefficient"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.32)))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    ax.barh(combined["feature"], combined["coefficient"], color=colors, alpha=0.8)
    ax.axvline(0, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linewidth=0.8)
    ax.set_xlabel("Lasso Coefficient", fontsize=11)
    ax.set_title(
        f"Top ±{top_n} N-gram Features (Lasso)\nTarget: {target_col}",
        fontsize=13,
    )
    style_axes(ax, grid_axis="x", grid_alpha=0.10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved coefficient plot → {output_path}")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    target_col: str = "target",
    tau: float = None,
) -> None:
    """Scatter plot of actual vs. cross-validated predicted values."""
    apply_spotify_theme()
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    ax.scatter(y_true, y_pred, alpha=0.55, s=25, color=SPOTIFY_COLORS.get("blue", "#4EA1FF"))

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, linestyle="--", linewidth=0.8, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), label="Perfect fit")

    tau_str = f"Kendall's τ = {tau:.3f}" if tau is not None else ""
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("CV Predicted", fontsize=11)
    ax.set_title(
        f"Actual vs. Predicted (5-fold CV)\n{target_col}  —  {tau_str}",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    style_axes(ax, grid_axis="both", grid_alpha=0.10)
    style_legend(ax)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved actual-vs-predicted plot → {output_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_lasso_text_analysis(
    sentences_path: str,
    doc_metrics_path: str,
    initiation_scores_path: Optional[str] = None,
    output_dir: str = "outputs/figures",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    cv: int = 5,
    compute_cv_predictions: bool = True,
) -> Dict[str, Dict]:
    """
    End-to-end N-gram Lasso pipeline.

    Targets:
      1. overall_kw_ai_ratio     — overall AI discussion intensity
      2. ai_initiation_score     — management-proactiveness score (if available)

    For each target, produces:
      - Volcano Plot
      - Coefficient bar chart
      - Actual vs. Predicted scatter with Kendall's Tau

    Args:
        sentences_path:        Parquet with sentence-level data.
        doc_metrics_path:      Parquet with document-level AI metrics.
        initiation_scores_path: Optional parquet with initiation scores.
        output_dir:            Directory for saving figures.

    Returns:
        dict mapping target_col → results dict from fit_lasso_ngram().
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading sentence data...")
    sentences_df = pd.read_parquet(sentences_path)

    print("Loading document metrics...")
    doc_metrics = pd.read_parquet(doc_metrics_path)

    # Build full-document corpus (speech + Q&A combined)
    print("Building document corpus...")
    corpus_df = _build_doc_corpus(sentences_df)
    print(f"  Documents in corpus: {len(corpus_df)}")

    # Build also speech-only and qa-only corpora
    speech_corpus = _build_doc_corpus(sentences_df, section="speech")
    qa_corpus     = _build_doc_corpus(sentences_df, section="qa")

    # Precompute TF-IDF/scaled matrices once per corpus variant (reused across targets)
    corpus_map = {
        "full": corpus_df,
        "speech": speech_corpus,
        "qa": qa_corpus,
    }
    precomputed_by_corpus: Dict[str, Dict] = {}

    def _get_precomputed(corpus_key: str) -> Dict:
        if corpus_key not in precomputed_by_corpus:
            cdf = corpus_map[corpus_key]
            print(
                f"Precomputing TF-IDF for {corpus_key} corpus "
                f"({len(cdf)} docs, max_features={max_features}, ngrams={ngram_range})..."
            )
            precomputed_by_corpus[corpus_key] = _precompute_corpus_features(
                cdf,
                max_features=max_features,
                ngram_range=ngram_range,
            )
        return precomputed_by_corpus[corpus_key]

    # Prepare target DataFrames
    targets = []

    # Target 1: overall AI ratio from doc_metrics
    if "overall_kw_ai_ratio" in doc_metrics.columns:
        targets.append(("overall_kw_ai_ratio", "full", doc_metrics))

    # Target 2: speech AI ratio — use speech-only corpus
    if "speech_kw_ai_ratio" in doc_metrics.columns:
        targets.append(("speech_kw_ai_ratio", "speech", doc_metrics))

    # Target 3: Q&A AI ratio — use Q&A-only corpus
    if "qa_kw_ai_ratio" in doc_metrics.columns:
        targets.append(("qa_kw_ai_ratio", "qa", doc_metrics))

    # Target 4: AI initiation score (management proactiveness)
    if initiation_scores_path and os.path.exists(initiation_scores_path):
        init_df = pd.read_parquet(initiation_scores_path)
        if "ai_initiation_score" in init_df.columns:
            if "total_ai_exchanges" in init_df.columns:
                before = len(init_df)
                init_df = init_df[init_df["total_ai_exchanges"].fillna(0) > 0].copy()
                removed = before - len(init_df)
                print(
                    f"Filtering initiation target to active AI exchanges only: "
                    f"removed {removed}, remaining {len(init_df)}"
                )
            if len(init_df) > 30:
                # Target is defined from Q&A exchanges; use the Q&A corpus for alignment.
                targets.append(("ai_initiation_score", "qa", init_df))
            else:
                print("Skipping ai_initiation_score: not enough active-AI rows after filtering.")
        else:
            print("Skipping ai_initiation_score: not enough data.")

    all_results = {}

    for target_col, corpus_key, target_df in targets:
        print(f"\n{'='*60}")
        print(f"Fitting Lasso for target: {target_col}")
        print("="*60)
        precomputed = None  # OOF path uses fold-local preprocessing; keep simple and robust.

        res = fit_lasso_ngram(
            corpus_df=corpus_map[corpus_key],
            target_df=target_df,
            target_col=target_col,
            max_features=max_features,
            ngram_range=ngram_range,
            cv=cv,
            precomputed_features=precomputed,
            compute_cv_predictions=compute_cv_predictions,
        )

        if not res:
            continue

        coef_df = res["coef_df"]
        tau     = res["kendall_tau"]
        y_true  = res["y_true"]
        y_pred  = res["y_pred"]

        safe_name = target_col.replace("/", "_")

        # Save coefficient table
        coef_csv = os.path.join(output_dir, f"lasso_coefs_{safe_name}.csv")
        coef_df.to_csv(coef_csv, index=False)
        print(f"Saved coefficient table → {coef_csv}")

        # Volcano Plot
        plot_volcano(
            coef_df,
            output_path=os.path.join(output_dir, f"volcano_{safe_name}.png"),
            target_col=target_col,
        )

        # Coefficient bar chart
        plot_top_coefficients(
            coef_df,
            output_path=os.path.join(output_dir, f"lasso_coef_bar_{safe_name}.png"),
            target_col=target_col,
        )

        # Actual vs. predicted
        if compute_cv_predictions and y_pred is not None:
            plot_actual_vs_predicted(
                y_true, y_pred,
                output_path=os.path.join(output_dir, f"lasso_fit_{safe_name}.png"),
                target_col=target_col,
                tau=tau,
            )

        all_results[target_col] = res

    # Summary table
    if all_results:
        summary_rows = []
        for tgt, r in all_results.items():
            summary_rows.append({
                "target": tgt,
                "n_docs": r["n_docs"],
                "alpha": round(r["alpha"], 6),
                "r2_train": round(r.get("r2_train", r["r2"]), 4),
                "r2_oof": None if r.get("r2_oof") is None else round(r["r2_oof"], 4),
                "kendall_tau": None if r["kendall_tau"] is None else round(r["kendall_tau"], 4),
                "kendall_p": None if r["kendall_p"] is None else round(r["kendall_p"], 4),
                "nonzero_features": len(r["coef_df"]),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "lasso_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved Lasso summary table → {summary_path}")
        print(summary_df.to_string(index=False))

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="N-gram Lasso + Volcano Plot")
    parser.add_argument(
        "--sentences",
        default="outputs/features/sentences_with_keywords.parquet",
        help="Path to sentence-level parquet with 'text', 'doc_id', 'section'.",
    )
    parser.add_argument(
        "--metrics",
        default="outputs/features/document_metrics.parquet",
        help="Path to document-level metrics parquet.",
    )
    parser.add_argument(
        "--initiation",
        default="outputs/features/initiation_scores.parquet",
        help="(Optional) Path to initiation scores parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures",
        help="Directory to save figures and CSVs.",
    )
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--skip-cv-pred",
        action="store_true",
        help="Skip outer cross-validated predictions/Kendall Tau scatter for faster runs.",
    )

    args = parser.parse_args()

    run_lasso_text_analysis(
        sentences_path=args.sentences,
        doc_metrics_path=args.metrics,
        initiation_scores_path=args.initiation,
        output_dir=args.output_dir,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        cv=args.cv,
        compute_cv_predictions=not args.skip_cv_pred,
    )
