from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.data.parse_transcripts import extract_rosters, split_prepared_and_qa


@dataclass(frozen=True)
class ParseOutcome:
    prepared_remarks_text: str
    qa_text: str
    qa_header_matched: bool
    pattern_used: str | None
    qa_start_idx: int
    prepared_len: int
    qa_len: int


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _build_call_id(ticker: str, datacqtr: str, earnings_date: str, transcript_raw: str) -> str:
    base = f"{ticker}|{datacqtr}|{earnings_date}"
    return f"{base}|{_stable_hash(transcript_raw)}"


def _iter_calls(df: pd.DataFrame) -> Iterable[dict[str, object]]:
    for _, row in df.iterrows():
        transcript_raw = str(row.get("transcript") or "")
        prepared, qa, diag = split_prepared_and_qa(transcript_raw)
        roster = extract_rosters(transcript_raw)

        yield {
            "ticker": row.get("ticker"),
            "datacqtr": row.get("datacqtr"),
            "earnings_date": str(row.get("earnings_date") or ""),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "transcript_raw": transcript_raw,
            "prepared_remarks_text": prepared,
            "qa_text": qa,
            "qa_header_matched": diag["qa_header_matched"],
            "pattern_used": diag["pattern_used"],
            "qa_start_idx": diag["qa_start_idx"],
            "prepared_len": diag["prepared_len"],
            "qa_len": diag["qa_len"],
            "roster_size": len(roster),
            "roster_map": roster,
        }


def build_calls_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = list(_iter_calls(df))
    calls = pd.DataFrame(rows)
    calls["call_id"] = calls.apply(
        lambda r: _build_call_id(
            str(r["ticker"]),
            str(r["datacqtr"]),
            str(r["earnings_date"]),
            str(r["transcript_raw"]),
        ),
        axis=1,
    )
    if not calls["call_id"].is_unique:
        calls["call_id"] = calls.apply(
            lambda r: _build_call_id(
                str(r["ticker"]),
                str(r["datacqtr"]),
                str(r["earnings_date"]),
                str(r["transcript_raw"]) + f"|{r.name}",
            ),
            axis=1,
        )
    return calls
