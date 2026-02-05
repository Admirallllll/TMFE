from __future__ import annotations

from src.features.ai_detection import apply_ai_rules


def test_ai_rule_thresholds():
    row = {"is_ai_kw": False, "ai_score_encoder": 0.9}
    out = apply_ai_rules(row, thr_hi=0.8, thr_lo=0.6)
    assert out["is_ai_final"] is True
    assert out["needs_review"] is False


def test_ai_rule_needs_review():
    row = {"is_ai_kw": False, "ai_score_encoder": 0.7}
    out = apply_ai_rules(row, thr_hi=0.8, thr_lo=0.6)
    assert out["is_ai_final"] is False
    assert out["needs_review"] is True


def test_ai_keyword_detection():
    from src.features.ai_detection import compute_is_ai_kw
    import pandas as pd

    s = pd.Series(["We use AI and machine learning.", "No mention here."])
    out = compute_is_ai_kw(s)
    assert out.tolist() == [True, False]
