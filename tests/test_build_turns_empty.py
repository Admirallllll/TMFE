from __future__ import annotations

import pandas as pd

from src.data.build_turns import build_turns_table


def test_build_turns_empty_schema():
    calls = pd.DataFrame(
        {
            "call_id": ["X|2024Q1|2024-01-01|hash"],
            "qa_text": [""],
        }
    )
    turns = build_turns_table(calls)
    assert "call_id" in turns.columns
    assert "turn_id" in turns.columns
    assert len(turns) == 0
