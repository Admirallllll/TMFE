"""Unit tests for regression data preparation and evaluation helpers."""

import pandas as pd
import numpy as np

from src.analysis.regression import (
    prepare_regression_data,
    _prepare_model_frame,
    compute_kendall_tau_oos,
    run_regression_analysis,
)


def test_prepare_regression_data_preserves_merge_keys(tmp_path):
    """Regression prep should keep parsed ticker/year/quarter for WRDS merge."""
    initiation = pd.DataFrame(
        [
            {
                "doc_id": "TEST_2024Q1",
                "ai_initiation_score": 0.7,
                "total_ai_exchanges": 2,
                "analyst_initiated_ratio": 0.5,
                "management_pivot_ratio": 0.5,
            }
        ]
    )
    doc_metrics = pd.DataFrame(
        [
            {
                "doc_id": "TEST_2024Q1",
                "speech_kw_ai_ratio": 0.1,
                "qa_kw_ai_ratio": 0.2,
                "overall_kw_ai_ratio": 0.15,
            }
        ]
    )
    wrds = pd.DataFrame(
        [
            {
                "tic": "TEST",
                "datacqtr": "2024Q1",
                "datadate": "2024-03-31",
                "xrdq": 10.0,
                "mkvaltq": 100.0,
                "prccq": 50.0,
                "epspxq": 1.2,
                "gsector": 45,
            }
        ]
    )

    initiation_path = tmp_path / "initiation_scores.parquet"
    doc_metrics_path = tmp_path / "document_metrics.parquet"
    wrds_path = tmp_path / "wrds.csv"
    initiation.to_parquet(initiation_path, index=False)
    doc_metrics.to_parquet(doc_metrics_path, index=False)
    wrds.to_csv(wrds_path, index=False)

    result = prepare_regression_data(str(initiation_path), str(doc_metrics_path), str(wrds_path))

    assert "ticker" in result.columns
    assert "year" in result.columns
    assert "quarter" in result.columns
    assert result.loc[0, "ticker"] == "TEST"
    assert result.loc[0, "year"] == 2024
    assert result.loc[0, "quarter"] == 1
    assert result.loc[0, "rd_intensity"] == 0.1


def test_prepare_model_frame_filters_no_ai_rows_for_initiation():
    df = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "ai_initiation_score": [0.5, 0.9, 0.2, 0.6],
            "total_ai_exchanges": [0, 2, 0, 3],
            "log_mktcap": [1.0, 2.0, 3.0, 4.0],
            "rd_intensity": [0.1, 0.2, 0.3, 0.4],
        }
    )
    model_df, attr = _prepare_model_frame(
        df,
        dv="ai_initiation_score",
        ivs=["log_mktcap", "rd_intensity"],
        filter_non_ai_initiation=True,
    )
    assert len(model_df) == 2
    assert (model_df["total_ai_exchanges"] > 0).all()
    assert attr["rows_removed_no_ai_filter"] == 2


def test_compute_kendall_tau_oos_returns_fold_predictions_grouped():
    rows = []
    for ticker_idx, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        for q in range(1, 4):
            rows.append(
                {
                    "ticker": ticker,
                    "ai_initiation_score": 0.1 * ticker_idx + 0.03 * q,
                    "log_mktcap": 10 + ticker_idx,
                    "rd_intensity": 0.01 * (q + ticker_idx),
                    "eps_positive": int((ticker_idx + q) % 2 == 0),
                }
            )
    df = pd.DataFrame(rows)
    metrics = compute_kendall_tau_oos(
        df,
        dv="ai_initiation_score",
        ivs=["log_mktcap", "rd_intensity", "eps_positive"],
        group_col="ticker",
        n_splits=3,
        random_state=42,
    )
    assert len(metrics["oof_predictions"]) == len(df)
    assert metrics["split_method"].startswith("GroupKFold")
    assert np.isfinite(metrics["kendall_tau"]) or np.isnan(metrics["kendall_tau"])


def test_run_regression_analysis_excludes_model4_and_writes_attrition(tmp_path):
    # Build synthetic aligned datasets.
    init_rows = []
    metric_rows = []
    wrds_rows = []
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for t_idx, ticker in enumerate(tickers):
        for q in range(1, 3):
            doc_id = f"{ticker}_2024Q{q}"
            total_ai = 0 if (q == 1 and t_idx % 2 == 0) else 2
            init_rows.append(
                {
                    "doc_id": doc_id,
                    "ai_initiation_score": 0.5 if total_ai == 0 else 0.2 + 0.1 * q + 0.05 * t_idx,
                    "total_ai_exchanges": total_ai,
                    "analyst_initiated_ratio": 0.4,
                    "management_pivot_ratio": 0.6,
                    "analyst_only_count": 0,
                    "analyst_initiated_count": 1,
                    "management_pivot_count": 1,
                }
            )
            metric_rows.append(
                {
                    "doc_id": doc_id,
                    "speech_kw_ai_ratio": 0.01 * (q + t_idx),
                    "qa_kw_ai_ratio": 0.02 * (q + 1),
                    "overall_kw_ai_ratio": 0.03 * (t_idx + 1),
                }
            )
            wrds_rows.append(
                {
                    "tic": ticker,
                    "datacqtr": f"2024Q{q}",
                    "datadate": f"2024-0{3*q}-28",
                    "xrdq": 10 + t_idx + q,
                    "mkvaltq": 100 + 10 * t_idx + q,
                    "prccq": 50 + t_idx,
                    "epspxq": 1.0 if (q + t_idx) % 2 == 0 else -1.0,
                    "gsector": 45,
                }
            )

    init_path = tmp_path / "initiation.parquet"
    doc_metrics_path = tmp_path / "doc_metrics.parquet"
    wrds_path = tmp_path / "wrds.csv"
    out_dir = tmp_path / "figs"
    out_dir.mkdir()

    pd.DataFrame(init_rows).to_parquet(init_path, index=False)
    pd.DataFrame(metric_rows).to_parquet(doc_metrics_path, index=False)
    pd.DataFrame(wrds_rows).to_csv(wrds_path, index=False)

    results = run_regression_analysis(
        str(init_path),
        str(doc_metrics_path),
        str(wrds_path),
        output_dir=str(out_dir),
        oos_cv_folds=3,
    )

    assert "model4" not in results
    assert "oos_metrics" in results
    assert (out_dir / "regression_sample_attrition.csv").exists()
    summary_text = (out_dir / "regression_summary.txt").read_text(encoding="utf-8")
    assert "Model 4" not in summary_text
