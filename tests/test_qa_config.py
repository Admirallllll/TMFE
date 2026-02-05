from __future__ import annotations

from src.config import QAConfig


def test_qa_config_constants():
    cfg = QAConfig()
    assert cfg.min_qa_found_rate_dev == 0.60
    assert cfg.min_qa_found_rate_full == 0.80
    assert cfg.min_assigned_char_pct_dev == 0.70
    assert cfg.min_assigned_char_pct_full == 0.80
    assert cfg.ai_thr_lo <= cfg.ai_thr_hi
