"""Unit tests for Lasso text feature analysis."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

import src.analysis.lasso_text_features as ltf


def _make_lasso_corpus(n_docs: int = 36):
    corpus_df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n_docs)],
            "text": [
                f"ai platform automation efficiency growth token{i % 5} margin signal{i % 3}"
                for i in range(n_docs)
            ],
        }
    )
    target_df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n_docs)],
            "target": np.linspace(0.0, 1.0, n_docs),
        }
    )
    return corpus_df, target_df


def test_build_lasso_text_pipeline_uses_sparse_safe_fast_defaults():
    pipe = ltf._build_lasso_text_pipeline(
        inner_cv=3,
        max_features=5000,
        ngram_range=(1, 2),
        random_state=42,
    )

    tfidf = pipe.named_steps["tfidf"]
    scaler = pipe.named_steps["scaler"]
    lasso = pipe.named_steps["lasso"]

    assert isinstance(pipe, SkPipeline)
    assert isinstance(scaler, StandardScaler)
    assert scaler.with_mean is False
    assert tfidf.max_features == 5000
    assert tfidf.min_df == 0.01
    assert tfidf.max_df == 0.8
    assert lasso.cv == 3
    assert lasso.n_jobs == -1
    assert lasso.n_alphas == 50
    assert lasso.max_iter == 1000
    assert lasso.tol == 1e-3


def test_fit_lasso_uses_pipeline_for_oof_predictions(monkeypatch):
    """OOF predictions should be produced by fold-local sklearn Pipelines."""
    corpus_df, target_df = _make_lasso_corpus()
    created = []
    real_pipeline = ltf.Pipeline

    def tracking_pipeline(*args, **kwargs):
        pipe = real_pipeline(*args, **kwargs)
        created.append(pipe)
        return pipe

    monkeypatch.setattr(ltf, "Pipeline", tracking_pipeline)

    result = ltf.fit_lasso_ngram(
        corpus_df=corpus_df,
        target_df=target_df,
        target_col="target",
        max_features=40,
        ngram_range=(1, 1),
        cv=3,
    )

    assert result
    assert any(isinstance(p, SkPipeline) for p in created)
    assert result["y_pred"] is not None
    assert len(result["y_pred"]) == len(result["y_true"])
    assert "kendall_tau_oof" in result


def test_fit_lasso_returns_coefficients_after_refactor():
    corpus_df, target_df = _make_lasso_corpus()
    result = ltf.fit_lasso_ngram(
        corpus_df=corpus_df,
        target_df=target_df,
        target_col="target",
        max_features=40,
        ngram_range=(1, 1),
        cv=2,
        compute_cv_predictions=False,
    )
    assert result
    assert "coef_df" in result
    assert "r2_train" in result
    assert result["coef_df"].shape[1] >= 3


def test_run_lasso_filters_initiation_target_and_uses_qa_corpus(tmp_path, monkeypatch):
    n_docs = 42
    sentences_rows = []
    doc_metrics_rows = []
    init_rows = []

    for i in range(n_docs):
        doc_id = f"TK{i:03d}_2024Q1"
        grp = i % 6
        sentences_rows.append(
            {"doc_id": doc_id, "section": "qa", "text": f"qa ai topic growth cluster{grp} signal{grp}"}
        )
        sentences_rows.append(
            {"doc_id": doc_id, "section": "speech", "text": f"speech remarks outlook cluster{grp}"}
        )
        doc_metrics_rows.append(
            {
                "doc_id": doc_id,
                "overall_kw_ai_ratio": 0.1 + 0.001 * i,
                "speech_kw_ai_ratio": 0.05 + 0.001 * i,
                "qa_kw_ai_ratio": 0.08 + 0.001 * i,
            }
        )
        total_ai = 0 if i < 10 else 2
        init_rows.append(
            {
                "doc_id": doc_id,
                "ai_initiation_score": 0.5 if total_ai == 0 else 0.2 + 0.01 * (i % 10),
                "total_ai_exchanges": total_ai,
            }
        )

    sentences_path = tmp_path / "sentences.parquet"
    metrics_path = tmp_path / "metrics.parquet"
    init_path = tmp_path / "initiation.parquet"
    out_dir = tmp_path / "figs"

    pd.DataFrame(sentences_rows).to_parquet(sentences_path, index=False)
    pd.DataFrame(doc_metrics_rows).to_parquet(metrics_path, index=False)
    pd.DataFrame(init_rows).to_parquet(init_path, index=False)

    calls = []

    def fake_fit_lasso_ngram(*, corpus_df, target_df, target_col, **kwargs):
        calls.append(
            {
                "target_col": target_col,
                "n_target": len(target_df),
                "n_corpus": len(corpus_df),
                "sections_present": set(),
            }
        )
        return {
            "coef_df": pd.DataFrame(
                {
                    "feature": ["ai"],
                    "coefficient": [0.1],
                    "doc_frequency": [10],
                    "log_doc_frequency": [np.log1p(10)],
                }
            ),
            "y_true": np.array([0.1, 0.2]),
            "y_pred": np.array([0.1, 0.2]),
            "alpha": 0.01,
            "r2": 0.1,
            "r2_train": 0.1,
            "r2_oof": 0.05,
            "kendall_tau": 0.2,
            "kendall_tau_oof": 0.2,
            "kendall_p": 0.5,
            "target_col": target_col,
            "n_docs": len(target_df),
            "vectorizer": None,
            "lasso": None,
        }

    monkeypatch.setattr(ltf, "fit_lasso_ngram", fake_fit_lasso_ngram)
    monkeypatch.setattr(ltf, "plot_volcano", lambda *args, **kwargs: None)
    monkeypatch.setattr(ltf, "plot_top_coefficients", lambda *args, **kwargs: None)
    monkeypatch.setattr(ltf, "plot_actual_vs_predicted", lambda *args, **kwargs: None)

    ltf.run_lasso_text_analysis(
        sentences_path=str(sentences_path),
        doc_metrics_path=str(metrics_path),
        initiation_scores_path=str(init_path),
        output_dir=str(out_dir),
        max_features=50,
        ngram_range=(1, 1),
        cv=2,
    )

    init_calls = [c for c in calls if c["target_col"] == "ai_initiation_score"]
    assert len(init_calls) == 1
    assert init_calls[0]["n_target"] == 32  # filtered out 10 neutral/no-AI rows
