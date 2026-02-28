"""Unit tests for EDA foundation aggregation helpers."""

import math

import pandas as pd

from src.analysis.eda_foundation import (
    compute_data_funnel,
    summarize_ratio_columns,
    compute_ai_exchange_zero_split,
)


def test_compute_data_funnel_counts_and_section_shares():
    parsed = pd.DataFrame({"doc_id": ["A_2024Q1", "B_2024Q1", "C_2024Q1"]})
    sentences = pd.DataFrame(
        {
            "doc_id": ["A_2024Q1"] * 3 + ["B_2024Q1"] * 2,
            "section": ["speech", "qa", "qa", "speech", "qa"],
            "text": ["a", "b", "c", "d", "e"],
        }
    )
    initiation = pd.DataFrame({"doc_id": ["A_2024Q1", "B_2024Q1"]})

    funnel = compute_data_funnel(parsed, sentences, initiation)

    assert funnel["total_parsed_documents"] == 3
    assert funnel["total_sentences"] == 5
    assert funnel["speech_sentences"] == 2
    assert funnel["qa_sentences"] == 3
    assert math.isclose(funnel["speech_sentence_share"], 0.4, rel_tol=1e-9)
    assert math.isclose(funnel["qa_sentence_share"], 0.6, rel_tol=1e-9)
    assert funnel["tracked_initiation_documents"] == 2


def test_summarize_ratio_columns_has_expected_statistics():
    df = pd.DataFrame(
        {
            "speech_kw_ai_ratio": [0.0, 0.0, 0.1, 0.2],
            "qa_kw_ai_ratio": [0.0, 0.05, 0.1, 0.2],
            "overall_kw_ai_ratio": [0.0, 0.02, 0.05, 0.1],
        }
    )

    summary = summarize_ratio_columns(df)

    assert set(summary["metric"]) == {
        "speech_kw_ai_ratio",
        "qa_kw_ai_ratio",
        "overall_kw_ai_ratio",
    }

    speech_row = summary.loc[summary["metric"] == "speech_kw_ai_ratio"].iloc[0]
    assert speech_row["median"] == 0.05
    assert speech_row["zero_share"] == 0.5


def test_compute_ai_exchange_zero_split():
    initiation = pd.DataFrame({"total_ai_exchanges": [0, 0, 1, 2, 0]})

    split = compute_ai_exchange_zero_split(initiation)

    assert split["total_documents"] == 5
    assert split["zero_ai_exchanges_count"] == 3
    assert split["nonzero_ai_exchanges_count"] == 2
    assert math.isclose(split["zero_ai_exchanges_share"], 0.6, rel_tol=1e-9)
    assert math.isclose(split["nonzero_ai_exchanges_share"], 0.4, rel_tol=1e-9)
