from __future__ import annotations

import pandas as pd

from src.config import QAConfig
from src.pipeline.validate_outputs import validate_tables


def test_validation_fails_empty():
    cfg = QAConfig()
    try:
        validate_tables(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            dev_mode=True,
            qa_cfg=cfg,
        )
        assert False
    except ValueError:
        assert True
