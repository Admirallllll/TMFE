from __future__ import annotations

import pandas as pd


def build_qa_pairs(turns_df: pd.DataFrame, calls_df: pd.DataFrame) -> pd.DataFrame:
    if turns_df.empty:
        return pd.DataFrame()

    calls_meta = calls_df.set_index("call_id")[
        ["ticker", "datacqtr", "earnings_date", "sector", "industry"]
    ]
    rows: list[dict[str, object]] = []
    for (call_id, question_id), grp in turns_df.groupby(["call_id", "question_id"]):
        if pd.isna(question_id):
            continue
        q_turns = grp.loc[grp["turn_type"] == "question"].sort_values("turn_id")
        if q_turns.empty:
            continue
        a_turns = grp.loc[grp["turn_type"] == "answer"].sort_values("turn_id")
        question_text = " ".join(q_turns["turn_text"].astype(str).tolist()).strip()
        answer_text = " ".join(a_turns["turn_text"].astype(str).tolist()).strip()
        answer_speakers = ", ".join(a_turns["speaker_name"].astype(str).tolist())

        meta = calls_meta.loc[call_id].to_dict()
        rows.append(
            {
                "call_id": call_id,
                "question_id": int(question_id),
                "question_speaker": q_turns.iloc[0]["speaker_name"],
                "question_text": question_text,
                "answer_text_concat": answer_text,
                "answer_turn_count": int(len(a_turns)),
                "answer_speakers": answer_speakers,
                "ticker": meta.get("ticker"),
                "datacqtr": meta.get("datacqtr"),
                "earnings_date": meta.get("earnings_date"),
                "sector": meta.get("sector"),
                "industry": meta.get("industry"),
            }
        )
    return pd.DataFrame(rows)
