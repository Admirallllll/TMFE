from __future__ import annotations

import pandas as pd

from src.analysis.introducer import ai_first_turn_position_distribution


def test_ai_first_turn_position_distribution():
    df = pd.DataFrame({"first_ai_turn_index_norm": [0.1, 0.2, 0.9, None]})
    out = ai_first_turn_position_distribution(df)
    assert out["n_calls"].sum() == 3
    assert set(out.columns) >= {"bucket", "n_calls"}
