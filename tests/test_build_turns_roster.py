from __future__ import annotations

import pandas as pd

from src.data.build_turns import build_turns_table


def test_build_turns_uses_roster_map():
    calls = pd.DataFrame(
        {
            "call_id": ["AAA|2024Q1|2024-02-01|hash"],
            "ticker": ["AAA"],
            "datacqtr": ["2024Q1"],
            "earnings_date": ["2024-02-01"],
            "sector": ["Tech"],
            "industry": ["Software"],
            "qa_text": ["John A. Doe: Welcome."],
            "roster_map": [{"john doe": "management"}],
        }
    )
    turns = build_turns_table(calls)
    assert len(turns) == 1
    assert turns.iloc[0]["speaker_role"] == "management"
    assert bool(turns.iloc[0]["roster_matched"]) is True
