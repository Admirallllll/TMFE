from __future__ import annotations

import pandas as pd


def call_level_introducer(turns_df: pd.DataFrame) -> pd.DataFrame:
    if turns_df.empty:
        return pd.DataFrame(columns=["call_id", "introduced_by", "first_ai_turn_index_norm", "first_ai_turn_type"])

    rows: list[dict[str, object]] = []
    for call_id, grp in turns_df.groupby("call_id"):
        grp_sorted = grp.sort_values("turn_id")
        ai_turns = grp_sorted.loc[grp_sorted["is_ai_final"]]
        if ai_turns.empty:
            rows.append(
                {
                    "call_id": call_id,
                    "introduced_by": "none",
                    "first_ai_turn_index_norm": None,
                    "first_ai_turn_type": None,
                }
            )
            continue
        first = ai_turns.iloc[0]
        total_turns = len(grp_sorted)
        rows.append(
            {
                "call_id": call_id,
                "introduced_by": first["speaker_role"] if first["speaker_role"] in {"analyst", "management"} else "other",
                "first_ai_turn_index_norm": float(first["turn_id"]) / float(total_turns) if total_turns > 0 else None,
                "first_ai_turn_type": first["turn_type"],
            }
        )
    return pd.DataFrame(rows)


def introduced_by_by_quarter(call_summary: pd.DataFrame, calls_df: pd.DataFrame) -> pd.DataFrame:
    if call_summary.empty:
        return pd.DataFrame(columns=["datacqtr", "introduced_by", "n_calls"])
    merged = call_summary.merge(calls_df[["call_id", "datacqtr"]], on="call_id", how="left")
    out = (
        merged.groupby(["datacqtr", "introduced_by"])
        .size()
        .rename("n_calls")
        .reset_index()
    )
    return out


def introduced_by_by_sector(call_summary: pd.DataFrame, calls_df: pd.DataFrame) -> pd.DataFrame:
    if call_summary.empty:
        return pd.DataFrame(columns=["sector", "introduced_by", "n_calls"])
    merged = call_summary.merge(calls_df[["call_id", "sector"]], on="call_id", how="left")
    out = (
        merged.groupby(["sector", "introduced_by"])
        .size()
        .rename("n_calls")
        .reset_index()
    )
    return out


def ai_first_turn_position_distribution(call_summary: pd.DataFrame) -> pd.DataFrame:
    if call_summary.empty:
        return pd.DataFrame(columns=["bucket", "n_calls"])
    vals = call_summary["first_ai_turn_index_norm"].dropna()
    if vals.empty:
        return pd.DataFrame(columns=["bucket", "n_calls"])
    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
    bucketed = pd.cut(vals.clip(0, 1), bins=bins, labels=labels, include_lowest=True)
    out = bucketed.value_counts().reindex(labels, fill_value=0).rename("n_calls").to_frame()
    out["bucket"] = out.index
    out = out.reset_index(drop=True)
    return out
