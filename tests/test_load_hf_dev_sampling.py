from __future__ import annotations

import pandas as pd

from src.data.load_hf import _sample_dev


def _make_df() -> pd.DataFrame:
    rows = []
    for t in ["A", "B", "C"]:
        for q in range(4):
            rows.append({"ticker": t, "datacqtr": f"2022Q{q+1}", "x": q})
    return pd.DataFrame(rows)


def test_sample_dev_keeps_full_ticker_groups():
    df = _make_df()
    out = _sample_dev(df, n=5, seed=7)
    assert len(out) >= 5
    for t in out["ticker"].unique():
        assert len(out.loc[out["ticker"] == t]) == len(df.loc[df["ticker"] == t])


def test_sample_dev_returns_all_when_n_large():
    df = _make_df()
    out = _sample_dev(df, n=len(df), seed=1)
    assert len(out) == len(df)
