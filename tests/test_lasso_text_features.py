"""Unit tests for Stage 12 text + sentiment forward R&D prediction."""

import numpy as np
import pandas as pd

import src.analysis.lasso_text_features as ltf


def test_compute_ai_sentiment_features_uses_only_ai_sentences():
    sentences = pd.DataFrame(
        {
            "doc_id": ["A_2024Q1", "A_2024Q1", "A_2024Q1", "B_2024Q1"],
            "kw_is_ai": [True, True, False, True],
            "text": [
                "ai efficiency gains and opportunity",
                "ai risk and uncertainty",
                "strong growth outside ai mention",
                "ai automation improves margin",
            ],
        }
    )

    feat = ltf.compute_ai_sentiment_features(sentences)
    row_a = feat.loc[feat["doc_id"] == "A_2024Q1"].iloc[0]
    row_b = feat.loc[feat["doc_id"] == "B_2024Q1"].iloc[0]

    assert np.isclose(row_a["ai_sentiment_positive_ratio"], 0.5)
    assert np.isclose(row_a["ai_sentiment_negative_ratio"], 0.5)
    assert np.isclose(row_b["ai_sentiment_positive_ratio"], 1.0)
    assert np.isclose(row_b["ai_sentiment_negative_ratio"], 0.0)


def test_fit_lasso_ngram_supports_binary_target_with_sentiment_features():
    n_docs = 40
    corpus_df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n_docs)],
            "text": [
                ("ai automation efficiency margin" if i % 2 == 0 else "ai hype risk uncertainty")
                for i in range(n_docs)
            ],
        }
    )
    target_df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n_docs)],
            "rd_increased_next_quarter": [1 if i % 2 == 0 else 0 for i in range(n_docs)],
        }
    )
    sentiment_df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n_docs)],
            "ai_sentiment_positive_ratio": [1.0 if i % 2 == 0 else 0.0 for i in range(n_docs)],
            "ai_sentiment_negative_ratio": [0.0 if i % 2 == 0 else 1.0 for i in range(n_docs)],
        }
    )

    result = ltf.fit_lasso_ngram(
        corpus_df=corpus_df,
        target_df=target_df,
        target_col="rd_increased_next_quarter",
        max_features=50,
        ngram_range=(1, 2),
        cv=3,
        extra_features_df=sentiment_df,
        task_type="classification",
    )

    assert result
    assert "coef_df" in result
    assert "metrics" in result
    assert set(["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]).issubset(result["metrics"].keys())


def test_run_lasso_outputs_forward_rd_prediction_artifacts(tmp_path):
    rows = []
    metrics_rows = []
    reg_rows = []
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for t_idx, ticker in enumerate(tickers):
        for q in [1, 2, 3, 4]:
            doc_id = f"{ticker}_2024Q{q}"
            is_pos = (t_idx + q) % 2 == 0
            rows.append(
                {
                    "doc_id": doc_id,
                    "section": "qa",
                    "kw_is_ai": True,
                    "text": "ai automation efficiency margin opportunity" if is_pos else "ai risk uncertainty pressure",
                }
            )
            rows.append(
                {
                    "doc_id": doc_id,
                    "section": "speech",
                    "kw_is_ai": False,
                    "text": "prepared remarks",
                }
            )
            metrics_rows.append({"doc_id": doc_id, "overall_kw_ai_ratio": 0.1, "speech_kw_ai_ratio": 0.05, "qa_kw_ai_ratio": 0.12})
            reg_rows.append(
                {
                    "doc_id": doc_id,
                    "ticker": ticker,
                    "year": 2024,
                    "quarter": q,
                    "rd_intensity": 0.02 + 0.002 * q + (0.003 if is_pos else 0.0),
                    "log_mktcap": 7.0 + 0.02 * t_idx,
                    "eps_positive": int(is_pos),
                }
            )

    sentences_path = tmp_path / "sentences_with_keywords.parquet"
    metrics_path = tmp_path / "document_metrics.parquet"
    reg_path = tmp_path / "regression_dataset.parquet"
    out_dir = tmp_path / "figs"
    out_dir.mkdir()

    pd.DataFrame(rows).to_parquet(sentences_path, index=False)
    pd.DataFrame(metrics_rows).to_parquet(metrics_path, index=False)
    pd.DataFrame(reg_rows).to_parquet(reg_path, index=False)

    ltf.run_lasso_text_analysis(
        sentences_path=str(sentences_path),
        doc_metrics_path=str(metrics_path),
        initiation_scores_path=None,
        output_dir=str(out_dir),
        max_features=80,
        ngram_range=(1, 2),
        cv=3,
        compute_cv_predictions=True,
    )

    summary_path = out_dir / "lasso_summary.csv"
    senti_path = out_dir / "ai_sentiment_features.csv"
    coef_path = out_dir / "lasso_coefs_rd_increased_next_quarter.csv"
    roc_path = out_dir / "lasso_roc_rd_increased_next_quarter.png"

    assert summary_path.exists()
    assert senti_path.exists()
    assert coef_path.exists()
    assert roc_path.exists()

    summary_df = pd.read_csv(summary_path)
    assert "target" in summary_df.columns
    assert "ROC-AUC" in summary_df.columns
