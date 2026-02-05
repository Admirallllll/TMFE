from __future__ import annotations

import pandas as pd

from src.data.parse_transcripts import extract_turns


def build_turns_table(calls_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "call_id",
        "ticker",
        "datacqtr",
        "earnings_date",
        "sector",
        "industry",
        "turn_id",
        "speaker_name",
        "speaker_role",
        "speaker_raw_header",
        "speaker_title",
        "turn_text",
        "turn_text_raw",
        "turn_type",
        "char_start",
        "char_end",
        "question_id",
        "answer_group_id",
        "roster_matched",
    ]
    rows: list[dict[str, object]] = []
    for _, row in calls_df.iterrows():
        call_id = row["call_id"]
        qa_text = row.get("qa_text") or ""
        roster_map = row.get("roster_map") or {}
        turns, _diag = extract_turns(qa_text, roster=roster_map)
        for t in turns:
            rows.append(
                {
                    "call_id": call_id,
                    "ticker": row.get("ticker"),
                    "datacqtr": row.get("datacqtr"),
                    "earnings_date": row.get("earnings_date"),
                    "sector": row.get("sector"),
                    "industry": row.get("industry"),
                    "turn_id": t.turn_id,
                    "speaker_name": t.speaker_name,
                    "speaker_role": t.speaker_role,
                    "speaker_raw_header": t.speaker_raw_header,
                    "speaker_title": None,
                    "turn_text": t.turn_text_clean,
                    "turn_text_raw": t.turn_text_raw,
                    "turn_type": t.turn_type,
                    "char_start": t.char_start,
                    "char_end": t.char_end,
                    "question_id": t.question_id,
                    "answer_group_id": t.answer_group_id,
                    "roster_matched": t.roster_matched,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)
