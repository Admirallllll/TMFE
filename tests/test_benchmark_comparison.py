import pandas as pd
import numpy as np

from src.analysis.benchmark_comparison import (
    evaluate_benchmark_models,
    write_benchmark_outputs,
)


def _make_synthetic_regression_df() -> pd.DataFrame:
    rows = []
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for t_idx, ticker in enumerate(tickers):
        for q in range(1, 5):
            doc_id = f"{ticker}_2024Q{q}"
            # Vary target by ticker and quarter so a mean baseline should not be optimal.
            y = 0.1 * t_idx + 0.03 * q
            rows.append(
                {
                    "doc_id": doc_id,
                    "ticker": ticker,
                    "ai_initiation_score": y,
                    "total_ai_exchanges": 2,
                    "log_mktcap": 10 + t_idx,
                    "rd_intensity": 0.01 * (q + t_idx),
                    "eps_positive": int((t_idx + q) % 2 == 0),
                    "overall_kw_ai_ratio": 0.02 * q,
                    "qa_kw_ai_ratio": 0.05 * q + 0.02 * t_idx,
                    "speech_kw_ai_ratio": 0.01 * (4 - q),
                    "year": 2024,
                    "quarter": q,
                    "sector": str((t_idx % 3) + 10),
                }
            )
    return pd.DataFrame(rows)


def _make_synthetic_sentences_df(reg_df: pd.DataFrame) -> pd.DataFrame:
    sentence_rows = []
    for _, row in reg_df.iterrows():
        n = int(row["quarter"]) + 1
        token = "alpha" if row["ticker"] in {"AAA", "BBB", "CCC"} else "beta"
        for i in range(n):
            sentence_rows.append(
                {
                    "doc_id": row["doc_id"],
                    "section": "qa",
                    "text": f"{token} ai strategy growth q{int(row['quarter'])} s{i}",
                }
            )
        # Add a speech sentence so section filtering behavior is exercised.
        sentence_rows.append(
            {
                "doc_id": row["doc_id"],
                "section": "speech",
                "text": "prepared remarks and outlook",
            }
        )
    return pd.DataFrame(sentence_rows)


def test_evaluate_benchmark_models_uses_group_splits_and_models_differ():
    reg_df = _make_synthetic_regression_df()
    sentences_df = _make_synthetic_sentences_df(reg_df)

    folds_df, summary_df = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=sentences_df,
        target_col="ai_initiation_score",
        group_col="ticker",
        n_splits=3,
        random_state=42,
        text_section="qa",
        text_max_features=100,
    )

    # Expected core outputs.
    assert set(["Fold", "Model", "MAE", "RMSE", "Kendall Tau", "n_train", "n_test"]).issubset(folds_df.columns)
    assert set(["Model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "Kendall Tau_mean", "Kendall Tau_std"]).issubset(summary_df.columns)
    assert folds_df["Model"].nunique() >= 3
    assert "Text TF-IDF Lasso" in set(folds_df["Model"])

    # Same split sizes across models within each fold (shared evaluation split).
    sizes_per_fold = folds_df.groupby("Fold")["n_test"].nunique()
    assert (sizes_per_fold == 1).all()

    # A group (ticker) should not appear in both train and test for the same fold.
    for overlap_count in folds_df["group_overlap_count"]:
        assert overlap_count == 0

    # At least one learned model should beat the mean baseline on MAE on this synthetic data.
    mean_mae = summary_df.loc[summary_df["Model"] == "Mean Baseline", "MAE_mean"].iloc[0]
    best_non_mean = summary_df.loc[summary_df["Model"] != "Mean Baseline", "MAE_mean"].min()
    assert best_non_mean < mean_mae

    # Summary ordering should prioritize Kendall Tau first.
    taus = summary_df["Kendall Tau_mean"].to_numpy()
    assert (taus[:-1] >= taus[1:] - 1e-12).all()


def test_write_benchmark_outputs_writes_flat_summary_csv(tmp_path):
    reg_df = _make_synthetic_regression_df()
    sentences_df = _make_synthetic_sentences_df(reg_df)

    folds_df, summary_df = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=sentences_df,
        n_splits=3,
        random_state=42,
        text_max_features=100,
    )

    out_dir = tmp_path / "figures"
    out_dir.mkdir()
    paths = write_benchmark_outputs(folds_df, summary_df, output_dir=str(out_dir))

    saved_summary = pd.read_csv(paths["summary_csv"])
    assert "Model" in saved_summary.columns
    assert not saved_summary["Model"].isna().any()
    assert "MAE_mean" in saved_summary.columns
    assert "Kendall Tau_mean" in saved_summary.columns


def test_evaluate_benchmark_filters_no_ai_rows_for_initiation():
    reg_df = _make_synthetic_regression_df()
    reg_df.loc[:3, "total_ai_exchanges"] = 0
    sentences_df = _make_synthetic_sentences_df(reg_df)

    folds_df, _ = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=sentences_df,
        target_col="ai_initiation_score",
        n_splits=3,
        random_state=42,
        text_model_mode="ratios",
        verbose=False,
    )
    assert "Text-Ratio Lasso" in set(folds_df["Model"])

    # Each fold's train+test size should reflect the filtered dataset.
    expected_n = int((reg_df["total_ai_exchanges"] > 0).sum())
    first_fold = folds_df.iloc[0]
    assert int(first_fold["n_train"] + first_fold["n_test"]) == expected_n
