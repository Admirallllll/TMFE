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
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        min_df=3,
        max_df=0.95,
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
    if precomputed_features is not None:
        doc_lookup = precomputed_features["doc_lookup"]
        merged = doc_lookup.merge(
            target_df[[doc_id_col, target_col]].dropna(),
            on=doc_id_col,
            how="inner",
        )
    else:
        merged = corpus_df[[doc_id_col, text_col]].merge(
            target_df[[doc_id_col, target_col]].dropna(), on=doc_id_col, how="inner"
        )
    if len(merged) < 30:
        print(f"[Lasso] Insufficient data for '{target_col}': only {len(merged)} documents.")
        return {}

    y = merged[target_col].values

    if precomputed_features is not None:
        row_idx = merged["_row_idx"].to_numpy(dtype=np.int32)
        X_scaled = precomputed_features["X_scaled"][row_idx]
        vectorizer = precomputed_features["vectorizer"]
        X_bin = (X_scaled > 0).astype(np.int8)
        doc_freq = _doc_frequencies(vectorizer, X_bin)
    else:
        texts = merged[text_col].fillna("").astype(str).tolist()

        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=3,
            max_df=0.95,
        )
        X = vectorizer.fit_transform(texts)

        # Binary matrix for document-frequency computation
        X_bin = (X > 0).astype(np.int8)
        doc_freq = _doc_frequencies(vectorizer, X_bin)

        # Features already normalized by TF-IDF; apply variance scaling only.
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)

    print(f"  Training LassoCV on {X_scaled.shape[0]} samples x {X_scaled.shape[1]} features...")
    # n_jobs=-1 sometimes causes deadlocks with huge matrices on Macs.
    # Set n_jobs=2 or 1 if it hangs. verbose=1 shows progress.
    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=2000, n_jobs=2, verbose=1)
    lasso.fit(X_scaled, y)

    alpha = lasso.alpha_
    r2 = lasso.score(X_scaled, y)

    # Cross-validated predictions for Kendall Tau
    y_pred = None
    tau = None
    p_val = None
    if compute_cv_predictions:
        print("  Calculating cross-validated predictions for Kendall Tau...")
        # Use a fixed-alpha Lasso in the outer CV loop to avoid nested LassoCV search.
        y_pred = cross_val_predict(
            Lasso(alpha=alpha, max_iter=2000),
            X_scaled,
            y,
            cv=cv,
        )
        tau, p_val = kendalltau(y, y_pred)

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
    tau_str = f"{tau:.4f}" if tau is not None else "skipped"
    p_str = f"{p_val:.4f}" if p_val is not None else "skipped"
    print(f"  Kendall's Tau:  {tau_str}  (p={p_str})")
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
        "kendall_tau": tau,
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

    df = coef_df.copy()
    df["color"] = df["coefficient"].apply(lambda c: "seagreen" if c > 0 else "crimson")
    df["abs_coef"] = df["coefficient"].abs()

    # Label the most impactful words on each side
    pos = df[df["coefficient"] > 0].nlargest(top_n_labels, "abs_coef")
    neg = df[df["coefficient"] < 0].nlargest(top_n_labels, "abs_coef")
    to_label = pd.concat([pos, neg])

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(
        df["coefficient"],
        df["log_doc_frequency"],
        c=df["color"],
        alpha=0.55,
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

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Lasso Coefficient\n← Lower AI Initiation         Higher AI Initiation →", fontsize=11)
    ax.set_ylabel("log(Document Frequency + 1)", fontsize=11)
    ax.set_title(
        f"Volcano Plot — N-gram Lasso Features\nTarget: {target_col}",
        fontsize=13,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen",
               markersize=8, label="Increases AI Initiation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
               markersize=8, label="Decreases AI Initiation"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
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

    pos = coef_df.nlargest(top_n, "coefficient")
    neg = coef_df.nsmallest(top_n, "coefficient")
    combined = pd.concat([neg, pos]).drop_duplicates("feature")
    combined = combined.sort_values("coefficient")

    colors = ["crimson" if c < 0 else "seagreen" for c in combined["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.32)))
    ax.barh(combined["feature"], combined["coefficient"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lasso Coefficient", fontsize=11)
    ax.set_title(
        f"Top ±{top_n} N-gram Features (Lasso)\nTarget: {target_col}",
        fontsize=13,
    )
    ax.grid(True, axis="x", alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved coefficient plot → {output_path}")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    target_col: str = "target",
    tau: float = None,
) -> None:
    """Scatter plot of actual vs. cross-validated predicted values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, s=25, color="steelblue")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="Perfect fit")

    tau_str = f"Kendall's τ = {tau:.3f}" if tau is not None else ""
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("CV Predicted", fontsize=11)
    ax.set_title(
        f"Actual vs. Predicted (5-fold CV)\n{target_col}  —  {tau_str}",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
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
        if "ai_initiation_score" in init_df.columns and len(init_df) > 30:
            targets.append(("ai_initiation_score", "full", init_df))
        else:
            print("Skipping ai_initiation_score: not enough data.")

    all_results = {}

    for target_col, corpus_key, target_df in targets:
        print(f"\n{'='*60}")
        print(f"Fitting Lasso for target: {target_col}")
        print("="*60)
        precomputed = _get_precomputed(corpus_key)

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
                "r2_train": round(r["r2"], 4),
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
