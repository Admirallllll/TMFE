from __future__ import annotations

import pandas as pd

from src.viz.qa_figures import plot_trend_line


def test_plot_raises_on_empty(tmp_path):
    df = pd.DataFrame()
    try:
        plot_trend_line(df, x_col="datacqtr", y_col="ai_question_rate", title="t", out_path=tmp_path / "x.png")
        assert False
    except ValueError:
        assert True
