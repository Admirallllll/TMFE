from __future__ import annotations

import pandas as pd


def answers_uncertainty_by_ai(turns_df: pd.DataFrame) -> pd.DataFrame:
    df = turns_df.loc[turns_df["turn_type"] == "answer"].copy()
    if df.empty:
        return pd.DataFrame(columns=["ai_bucket", "uncertainty_mean", "n_answers"])
    df["ai_bucket"] = df["is_ai_final"].map({True: "ai", False: "non_ai"})
    out = (
        df.groupby("ai_bucket")["lm_unc_per1k"]
        .mean()
        .rename("uncertainty_mean")
        .to_frame()
    )
    out["n_answers"] = df.groupby("ai_bucket").size().astype(int)
    out = out.reset_index()
    return out


def answers_uncertainty_by_introducer(turns_df: pd.DataFrame, call_summary: pd.DataFrame) -> pd.DataFrame:
    if turns_df.empty or call_summary.empty:
        return pd.DataFrame(columns=["introduced_by", "uncertainty_mean", "n_answers"])
    merged = turns_df.merge(call_summary[["call_id", "introduced_by"]], on="call_id", how="left")
    merged = merged.loc[merged["turn_type"] == "answer"].copy()
    if merged.empty:
        return pd.DataFrame(columns=["introduced_by", "uncertainty_mean", "n_answers"])
    out = (
        merged.groupby("introduced_by")["lm_unc_per1k"]
        .mean()
        .rename("uncertainty_mean")
        .to_frame()
    )
    out["n_answers"] = merged.groupby("introduced_by").size().astype(int)
    out = out.reset_index()
    return out
