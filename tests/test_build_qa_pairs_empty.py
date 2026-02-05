from __future__ import annotations

import pandas as pd

from src.data.build_qa_pairs import build_qa_pairs


def test_build_qa_pairs_empty_schema():
    turns = pd.DataFrame()
    calls = pd.DataFrame({"call_id": []})
    out = build_qa_pairs(turns, calls)
    assert "call_id" in out.columns
    assert "question_id" in out.columns
    assert len(out) == 0
