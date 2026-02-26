"""
Unit tests for regression data preparation.
"""

import pandas as pd

from src.analysis.regression import prepare_regression_data


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

    result = prepare_regression_data(
        str(initiation_path),
        str(doc_metrics_path),
        str(wrds_path),
    )

    assert "ticker" in result.columns
    assert "year" in result.columns
    assert "quarter" in result.columns
    assert result.loc[0, "ticker"] == "TEST"
    assert result.loc[0, "year"] == 2024
    assert result.loc[0, "quarter"] == 1
    assert result.loc[0, "rd_intensity"] == 0.1
