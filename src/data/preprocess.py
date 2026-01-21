from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.text import clean_text_basic, count_tokens


NUMERIC_COLUMNS: tuple[str, ...] = (
    "eps12mtrailing_qavg",
    "eps12mtrailing_eoq",
    "eps12mfwd_qavg",
    "eps12mfwd_eoq",
    "eps_lt",
    "peforw_qavg",
    "peforw_eoq",
)


@dataclass(frozen=True)
class PreprocessResult:
    df: pd.DataFrame
    dropped_missing_peforw_qavg: int


def preprocess_base_table(df: pd.DataFrame, *, remove_boilerplate: bool, logger) -> PreprocessResult:
    df = df.copy()

    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce", utc=True).dt.tz_convert(None)

    datacqtr_parts = df["datacqtr"].astype(str).str.extract(r"^(?P<datacqtr_year>\d{4})Q(?P<datacqtr_quarter>[1-4])$")
    df["datacqtr_year"] = pd.to_numeric(datacqtr_parts["datacqtr_year"], errors="coerce").astype("Int64")
    df["datacqtr_quarter"] = pd.to_numeric(datacqtr_parts["datacqtr_quarter"], errors="coerce").astype("Int64")
    df["quarter_index"] = (df["datacqtr_year"] * 4 + (df["datacqtr_quarter"] - 1)).astype("Int64")

    before = len(df)
    df = df.loc[df["quarter_index"].notna() & df["ticker"].notna()].reset_index(drop=True)
    dropped_bad_time = before - len(df)
    if dropped_bad_time:
        logger.info(f"Dropped {dropped_bad_time:,} rows with invalid datacqtr or missing ticker")

    df["transcript"] = df["transcript"].astype(str)
    df["clean_transcript"] = df["transcript"].map(clean_text_basic)
    if remove_boilerplate:
        df["clean_transcript"] = df["clean_transcript"].str.replace(r"\s+", " ", regex=True).str.strip()

    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}_missing"] = df[c].isna().astype("int8")

    before = len(df)
    df = df.loc[df["peforw_qavg"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped:,} rows with missing peforw_qavg")

    df["n_chars"] = df["clean_transcript"].str.len().fillna(0).astype("int64")
    df["n_tokens"] = df["clean_transcript"].map(count_tokens).astype("int64")

    return PreprocessResult(df=df, dropped_missing_peforw_qavg=dropped)
