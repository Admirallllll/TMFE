import numpy as np
import pandas as pd

from src.analysis.benchmark_comparison import evaluate_benchmark_models, write_benchmark_outputs


def _make_classification_df() -> pd.DataFrame:
    rows = []
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    for t_idx, ticker in enumerate(tickers):
        sector = "10" if t_idx < 4 else "20"
        for quarter in [1, 2, 3, 4]:
            mkvaltq = 1000 + 40 * t_idx + 20 * quarter
            rows.append(
                {
                    "doc_id": f"{ticker}_2024Q{quarter}",
                    "ticker": ticker,
                    "year": 2024,
                    "quarter": quarter,
                    "sector": sector,
                    "mkvaltq": mkvaltq,
                    "rd_intensity": 0.02 + 0.003 * (t_idx % 4) + 0.001 * quarter,
                    "log_mktcap": np.log(mkvaltq),
                    "eps_positive": int((t_idx + quarter) % 2 == 0),
                    "stock_price": 30 + 2 * t_idx + quarter,
                    "overall_kw_ai_ratio": 0.01 * quarter + 0.003 * t_idx,
                    "qa_kw_ai_ratio": 0.02 * quarter + 0.002 * t_idx,
                    "speech_kw_ai_ratio": 0.01 * (5 - quarter),
                }
            )
    return pd.DataFrame(rows)


def _make_sentences(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        token = "automation margin" if r["sector"] == "10" else "product growth"
        rows.append({"doc_id": r["doc_id"], "section": "qa", "text": f"ai {token} roadmap"})
        rows.append({"doc_id": r["doc_id"], "section": "speech", "text": "prepared remarks"})
    return pd.DataFrame(rows)


def test_benchmark_classification_metrics_and_group_split():
    reg_df = _make_classification_df()
    sentences_df = _make_sentences(reg_df)

    folds_df, summary_df = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=sentences_df,
        target_col="beats_sector_median",
        group_col="ticker",
        n_splits=4,
        text_model_mode="ratios",
        verbose=False,
    )

    expected_fold_cols = {
        "Fold",
        "Model",
        "ROC-AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "n_train",
        "n_test",
        "group_overlap_count",
        "target_source",
    }
    expected_summary_cols = {
        "Model",
        "ROC-AUC_mean",
        "Accuracy_mean",
        "Precision_mean",
        "Recall_mean",
        "F1-Score_mean",
    }

    assert expected_fold_cols.issubset(set(folds_df.columns))
    assert expected_summary_cols.issubset(set(summary_df.columns))
    assert folds_df["Model"].nunique() >= 3
    assert "target_growth_vs_sector_median" in set(folds_df["target_source"])
    assert (folds_df["group_overlap_count"] == 0).all()

    aucs = summary_df["ROC-AUC_mean"].to_numpy(dtype=float)
    assert (aucs[:-1] >= aucs[1:] - 1e-12).all()


def test_write_benchmark_outputs_writes_classification_summary(tmp_path):
    reg_df = _make_classification_df()
    folds_df, summary_df = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=None,
        target_col="beats_sector_median",
        n_splits=4,
        text_model_mode="ratios",
        verbose=False,
    )

    out_dir = tmp_path / "figures"
    out_dir.mkdir()
    paths = write_benchmark_outputs(folds_df, summary_df, output_dir=str(out_dir))

    saved_summary = pd.read_csv(paths["summary_csv"])
    assert "Model" in saved_summary.columns
    assert "ROC-AUC_mean" in saved_summary.columns
    assert "F1-Score_mean" in saved_summary.columns
    assert (out_dir / "benchmark_comparison_roc.png").exists()


def test_benchmark_target_falls_back_to_eps_sign():
    reg_df = _make_classification_df().drop(columns=["mkvaltq"]).copy()
    reg_df["y_next_eps_growth_yoy"] = np.where((reg_df["quarter"] % 2) == 0, 0.05, -0.02)

    folds_df, _ = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=None,
        target_col="beats_sector_median",
        n_splits=4,
        text_model_mode="ratios",
        verbose=False,
    )
    assert "target_eps_growth_positive" in set(folds_df["target_source"])
