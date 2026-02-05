from __future__ import annotations

import pandas as pd


def ai_question_rate_by_quarter(turns_df: pd.DataFrame) -> pd.DataFrame:
    df = turns_df.loc[turns_df["turn_type"] == "question"].copy()
    if df.empty:
        return pd.DataFrame(columns=["datacqtr", "ai_question_rate", "n_questions"])
    grouped = df.groupby("datacqtr")
    out = grouped["is_ai_final"].mean().rename("ai_question_rate").to_frame()
    out["n_questions"] = grouped.size().astype(int)
    out = out.reset_index()
    return out


def ai_answer_rate_by_quarter(turns_df: pd.DataFrame) -> pd.DataFrame:
    df = turns_df.loc[turns_df["turn_type"] == "answer"].copy()
    if df.empty:
        return pd.DataFrame(columns=["datacqtr", "ai_answer_rate", "n_answers"])
    grouped = df.groupby("datacqtr")
    out = grouped["is_ai_final"].mean().rename("ai_answer_rate").to_frame()
    out["n_answers"] = grouped.size().astype(int)
    out = out.reset_index()
    return out


def ai_questions_by_sector_quarter(turns_df: pd.DataFrame) -> pd.DataFrame:
    df = turns_df.loc[(turns_df["turn_type"] == "question") & (turns_df["is_ai_final"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["sector", "datacqtr", "ai_question_count"])
    out = (
        df.groupby(["sector", "datacqtr"])
        .size()
        .rename("ai_question_count")
        .reset_index()
    )
    return out
