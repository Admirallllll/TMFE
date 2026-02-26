"""
Unit tests for Lasso text feature analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

import src.analysis.lasso_text_features as ltf


def test_fit_lasso_uses_fixed_lasso_for_cv_predictions(monkeypatch):
    """Outer CV predictions should not trigger a nested LassoCV search."""
    n_docs = 36
    corpus_df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n_docs)],
            "text": [
                f"ai platform automation efficiency growth token{i % 5} "
                f"margin signal{i % 3}"
                for i in range(n_docs)
            ],
        }
    )
    target_df = pd.DataFrame(
        {
            "doc_id": [f"D{i:03d}" for i in range(n_docs)],
            "target": np.linspace(0.0, 1.0, n_docs),
        }
    )

    captured = {}

    def fake_cross_val_predict(estimator, X, y, cv):
        captured["estimator"] = estimator
        return np.zeros(len(y), dtype=float)

    monkeypatch.setattr(ltf, "cross_val_predict", fake_cross_val_predict)

    result = ltf.fit_lasso_ngram(
        corpus_df=corpus_df,
        target_df=target_df,
        target_col="target",
        max_features=40,
        ngram_range=(1, 1),
        cv=2,
    )

    assert result
    assert isinstance(captured["estimator"], Lasso)
