from __future__ import annotations

import pandas as pd

from src.analysis.trends import ai_question_rate_by_quarter


def test_ai_question_rate_nonempty():
    df = pd.DataFrame(
        {
            "datacqtr": ["2024Q1", "2024Q1", "2024Q2"],
            "turn_type": ["question", "question", "question"],
            "is_ai_final": [True, False, True],
        }
    )
    out = ai_question_rate_by_quarter(df)
    assert len(out) == 2
    assert set(out.columns) >= {"datacqtr", "ai_question_rate", "n_questions"}
