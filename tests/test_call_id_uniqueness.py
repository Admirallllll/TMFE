from __future__ import annotations

import pandas as pd

from src.data.build_calls import build_calls_table


def test_call_id_unique_with_hash_fallback():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "datacqtr": ["2024Q1", "2024Q1"],
            "earnings_date": ["2024-02-01", "2024-02-01"],
            "sector": ["Tech", "Tech"],
            "industry": ["Software", "Software"],
            "transcript": ["Call one text", "Call two text"],
        }
    )
    calls = build_calls_table(df)
    assert calls["call_id"].is_unique
