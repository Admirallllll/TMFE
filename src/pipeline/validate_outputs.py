"""Validate QA pipeline outputs for completeness and correctness."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import Paths, QAConfig


REQUIRED_FIGURES: tuple[str, ...] = (
    "trend_ai_question_rate_by_quarter.png",
    "trend_ai_answer_rate_by_quarter.png",
    "who_introduces_ai_first_by_quarter.png",
    "who_introduces_ai_first_by_sector.png",
    "ai_first_turn_position_distribution.png",
    "answers_uncertainty_ai_vs_nonai.png",
    "answers_uncertainty_analystfirst_vs_mgmtfirst.png",
)

OPTIONAL_FIGURES: tuple[str, ...] = (
    "ai_question_subtopics_trend.png",
)


def _summarize_df(df: pd.DataFrame) -> str:
    null_rates = df.isna().mean().sort_values(ascending=False)
    top_nulls = null_rates.head(5).to_dict()
    return f"shape={df.shape}, null_rates_top5={top_nulls}, head=\n{df.head(3).to_string(index=False)}"


def _require_nonempty(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        raise ValueError(f"{name} is empty. {_summarize_df(df)}")


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. {_summarize_df(df)}")


def validate_tables(
    calls_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    qa_pairs_df: pd.DataFrame,
    call_summary_df: pd.DataFrame,
    *,
    dev_mode: bool,
    qa_cfg: QAConfig,
) -> None:
    _require_nonempty(calls_df, "calls")
    _require_nonempty(turns_df, "turns")
    _require_nonempty(qa_pairs_df, "qa_pairs")
    _require_nonempty(call_summary_df, "call_summary")

    _require_columns(
        calls_df,
        [
            "call_id",
            "ticker",
            "datacqtr",
            "earnings_date",
            "sector",
            "industry",
            "transcript_raw",
            "prepared_remarks_text",
            "qa_text",
            "qa_header_matched",
            "pattern_used",
            "qa_start_idx",
            "prepared_len",
            "qa_len",
            "assigned_char_pct",
            "turn_count",
            "role_unresolved_rate",
            "roster_match_rate",
        ],
        "calls",
    )
    _require_columns(
        turns_df,
        [
            "call_id",
            "turn_id",
            "turn_type",
            "speaker_role",
            "turn_text",
            "is_ai_kw",
            "ai_score_encoder",
            "is_ai_final",
            "roster_matched",
        ],
        "turns",
    )
    _require_columns(
        qa_pairs_df,
        ["call_id", "question_id", "question_text", "answer_text_concat", "answer_turn_count"],
        "qa_pairs",
    )
    _require_columns(
        call_summary_df,
        ["call_id", "introduced_by", "first_ai_turn_index_norm", "first_ai_turn_type"],
        "call_summary",
    )

    if calls_df["assigned_char_pct"].isna().all():
        raise ValueError("calls.assigned_char_pct is all NaN. " + _summarize_df(calls_df))

    qa_found_rate = float(calls_df["qa_header_matched"].mean())
    assigned_char_mean = float(calls_df["assigned_char_pct"].mean())

    min_qa = qa_cfg.min_qa_found_rate_dev if dev_mode else qa_cfg.min_qa_found_rate_full
    min_char = qa_cfg.min_assigned_char_pct_dev if dev_mode else qa_cfg.min_assigned_char_pct_full

    if qa_found_rate < min_qa:
        msg = f"qa_header_matched rate {qa_found_rate:.2%} below threshold {min_qa:.2%}"
        if dev_mode:
            print("WARN:", msg)
        else:
            raise ValueError(msg)

    if assigned_char_mean < min_char:
        msg = f"assigned_char_pct mean {assigned_char_mean:.2%} below threshold {min_char:.2%}"
        if dev_mode:
            print("WARN:", msg)
        else:
            raise ValueError(msg)

    if call_summary_df["introduced_by"].isna().all():
        raise ValueError("introduced_by is all NaN. " + _summarize_df(call_summary_df))


def _validate_table_file(path: Path, *, required_cols: list[str], must_have_cols: list[str], name: str) -> None:
    if not path.exists():
        raise ValueError(f"Missing table: {path}")
    df = pd.read_csv(path)
    _require_nonempty(df, name)
    _require_columns(df, required_cols, name)
    for col in must_have_cols:
        if col in df.columns and df[col].isna().all():
            raise ValueError(f"{name} column {col} is all NaN. {_summarize_df(df)}")


def validate_figures(fig_dir: Path, results_md: Path) -> None:
    missing: list[str] = []
    for fig in REQUIRED_FIGURES:
        path = fig_dir / fig
        if not path.exists():
            missing.append(fig)
        elif path.stat().st_size < 2000:
            raise ValueError(f"Figure too small: {fig} ({path.stat().st_size} bytes)")

    for fig in OPTIONAL_FIGURES:
        path = fig_dir / fig
        if not path.exists():
            if results_md.exists():
                text = results_md.read_text(encoding="utf-8")
                if f"SKIPPED: {fig}" not in text:
                    raise ValueError(f"Optional figure missing without skip note in RESULTS.md: {fig}")
            else:
                raise ValueError(f"Optional figure missing and RESULTS.md not found: {fig}")

    if missing:
        raise ValueError(f"Missing required figures: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA pipeline outputs")
    parser.add_argument("--dev", action="store_true", help="Check dev outputs")
    args = parser.parse_args()

    paths = Paths()
    qa_cfg = QAConfig()
    subdir = "dev" if args.dev else ""

    calls_path = paths.processed_dir / subdir / "calls.parquet"
    turns_path = paths.processed_dir / subdir / "turns.parquet"
    qa_pairs_path = paths.processed_dir / subdir / "qa_pairs.parquet"
    call_summary_path = paths.processed_dir / subdir / "analysis_ready.parquet"

    calls = pd.read_parquet(calls_path)
    turns = pd.read_parquet(turns_path)
    qa_pairs = pd.read_parquet(qa_pairs_path)
    call_summary = pd.read_parquet(call_summary_path)

    validate_tables(calls, turns, qa_pairs, call_summary, dev_mode=args.dev, qa_cfg=qa_cfg)

    tables_dir = paths.tables_dir / subdir
    _validate_table_file(
        tables_dir / "ai_question_rate_by_quarter.csv",
        required_cols=["datacqtr", "ai_question_rate", "n_questions"],
        must_have_cols=["ai_question_rate"],
        name="ai_question_rate_by_quarter",
    )
    _validate_table_file(
        tables_dir / "ai_answer_rate_by_quarter.csv",
        required_cols=["datacqtr", "ai_answer_rate", "n_answers"],
        must_have_cols=["ai_answer_rate"],
        name="ai_answer_rate_by_quarter",
    )
    _validate_table_file(
        tables_dir / "who_introduces_ai_first_by_quarter.csv",
        required_cols=["datacqtr", "introduced_by", "n_calls"],
        must_have_cols=["n_calls"],
        name="who_introduces_ai_first_by_quarter",
    )
    _validate_table_file(
        tables_dir / "who_introduces_ai_first_by_sector.csv",
        required_cols=["sector", "introduced_by", "n_calls"],
        must_have_cols=["n_calls"],
        name="who_introduces_ai_first_by_sector",
    )
    _validate_table_file(
        tables_dir / "ai_first_turn_position_distribution.csv",
        required_cols=["bucket", "n_calls"],
        must_have_cols=["n_calls"],
        name="ai_first_turn_position_distribution",
    )
    _validate_table_file(
        tables_dir / "answers_uncertainty_ai_vs_nonai.csv",
        required_cols=["ai_bucket", "uncertainty_mean", "n_answers"],
        must_have_cols=["uncertainty_mean"],
        name="answers_uncertainty_ai_vs_nonai",
    )
    _validate_table_file(
        tables_dir / "answers_uncertainty_analystfirst_vs_mgmtfirst.csv",
        required_cols=["introduced_by", "uncertainty_mean", "n_answers"],
        must_have_cols=["uncertainty_mean"],
        name="answers_uncertainty_analystfirst_vs_mgmtfirst",
    )

    figures_dir = paths.figures_dir / subdir
    results_md = paths.outputs_dir / "RESULTS.md"
    validate_figures(figures_dir, results_md)

    print("All validations passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
