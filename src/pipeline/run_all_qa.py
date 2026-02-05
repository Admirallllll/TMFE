from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DatasetConfig, FeatureConfig, Paths, QAConfig, RunConfig
from src.data.build_calls import build_calls_table
from src.data.build_qa_pairs import build_qa_pairs
from src.data.build_turns import build_turns_table
from src.data.load_hf import load_hf_dataset_to_df
from src.features.ai_detection import apply_ai_rules, compute_is_ai_kw
from src.features.sentiment_lm import compute_lm_features
from src.utils.io import read_parquet, write_csv, write_parquet
from src.utils.logging import setup_logging
from src.utils.text import count_tokens
from src.analysis.trends import (
    ai_answer_rate_by_quarter,
    ai_question_rate_by_quarter,
    ai_questions_by_sector_quarter,
)
from src.analysis.introducer import (
    ai_first_turn_position_distribution,
    call_level_introducer,
    introduced_by_by_quarter,
    introduced_by_by_sector,
)
from src.analysis.uncertainty import (
    answers_uncertainty_by_ai,
    answers_uncertainty_by_introducer,
)
from src.pipeline.validate_outputs import validate_tables
from src.viz.qa_figures import plot_bar, plot_placeholder, plot_stacked_bar, plot_trend_line


def _artifact_paths(paths: Paths, *, dev_mode: bool) -> dict[str, Path]:
    subdir = "dev" if dev_mode else ""
    processed = paths.processed_dir / subdir
    processed.mkdir(parents=True, exist_ok=True)
    raw_path = paths.data_dir / ("raw_snapshot.parquet" if not dev_mode else "processed/dev/raw_snapshot.parquet")
    return {
        "raw": raw_path,
        "calls": processed / "calls.parquet",
        "turns": processed / "turns.parquet",
        "qa_pairs": processed / "qa_pairs.parquet",
        "analysis": processed / "analysis_ready.parquet",
    }


def _maybe_load(path: Path, *, recompute: bool) -> pd.DataFrame | None:
    if (not recompute) and path.exists():
        return read_parquet(path)
    return None


def _compute_assigned_char_pct(calls: pd.DataFrame, turns: pd.DataFrame) -> pd.Series:
    turn_chars = turns.groupby("call_id")["turn_text_raw"].apply(lambda s: s.fillna("").str.len().sum())
    qa_len = calls.set_index("call_id")["qa_len"].replace(0, np.nan)
    assigned = (turn_chars / qa_len).reindex(calls["call_id"]).fillna(0.0)
    return assigned


def _compute_role_rates(turns: pd.DataFrame, calls: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    by_call = turns.groupby("call_id")
    role_unresolved = by_call["speaker_role"].apply(lambda s: (s == "other").mean())
    roster_match = by_call["roster_matched"].mean()
    return (
        role_unresolved.reindex(calls["call_id"]).fillna(0.0),
        roster_match.reindex(calls["call_id"]).fillna(0.0),
    )


def _maybe_transfer_scores(turns: pd.DataFrame, *, cfg: FeatureConfig, paths: Paths, logger) -> pd.Series:
    encoder_dir = paths.transfer_model_dir / "encoder"
    config_path = encoder_dir / "transfer_config.json"
    if not (encoder_dir.exists() and config_path.exists()):
        logger.info("Transfer encoder not found; using zero ai_score_encoder")
        return pd.Series(np.zeros(len(turns), dtype="float64"))
    try:
        from src.models import train_transfer_encoder as te
        from src.features.transfer_inference import compute_transfer_features

        bundle = te._load_bundle(encoder_dir, config_path, cfg, logger=logger)
        tmp = turns[["turn_text"]].rename(columns={"turn_text": "clean_transcript"})
        feats = compute_transfer_features(tmp, encoder_bundle=bundle, cfg=cfg, logger=logger).features
        return feats["transfer_ai_prob"].astype("float64")
    except Exception as e:
        logger.info(f"Transfer encoder load/inference failed ({e}); using zero ai_score_encoder")
        return pd.Series(np.zeros(len(turns), dtype="float64"))


def run_all_qa(run_cfg: RunConfig, *, dataset_cfg: DatasetConfig, feature_cfg: FeatureConfig, qa_cfg: QAConfig) -> None:
    paths = Paths()
    artifacts = _artifact_paths(paths, dev_mode=run_cfg.dev_mode)

    fig_dir = paths.figures_dir / ("dev" if run_cfg.dev_mode else "")
    tbl_dir = paths.tables_dir / ("dev" if run_cfg.dev_mode else "")
    log_path = paths.logs_dir / ("qa_pipeline_dev.log" if run_cfg.dev_mode else "qa_pipeline.log")
    logger = setup_logging(log_path)

    for d in (fig_dir, tbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw = _maybe_load(artifacts["raw"], recompute=run_cfg.recompute)
    if raw is None:
        raw = load_hf_dataset_to_df(
            dataset_cfg.hf_name,
            dataset_cfg.hf_split,
            paths.hf_cache_dir,
            dev_mode=run_cfg.dev_mode,
            dev_sample_n=run_cfg.dev_sample_n,
            seed=run_cfg.seed,
            logger=logger,
            start_year=dataset_cfg.start_year,
        )
        write_parquet(raw, artifacts["raw"])
        logger.info(f"Saved raw snapshot: {artifacts['raw']}")

    calls = _maybe_load(artifacts["calls"], recompute=run_cfg.recompute)
    if calls is None:
        calls = build_calls_table(raw)
        write_parquet(calls.drop(columns=["roster_map"], errors="ignore"), artifacts["calls"])
        logger.info(f"Saved calls: {artifacts['calls']}")

    turns = _maybe_load(artifacts["turns"], recompute=run_cfg.recompute)
    if turns is None:
        if "roster_map" not in calls.columns:
            calls = build_calls_table(raw)
        turns = build_turns_table(calls)
        write_parquet(turns, artifacts["turns"])
        logger.info(f"Saved turns: {artifacts['turns']}")

    qa_pairs = _maybe_load(artifacts["qa_pairs"], recompute=run_cfg.recompute)
    if qa_pairs is None:
        qa_pairs = build_qa_pairs(turns, calls)
        write_parquet(qa_pairs, artifacts["qa_pairs"])
        logger.info(f"Saved qa_pairs: {artifacts['qa_pairs']}")

    calls = calls.copy()
    calls["turn_count"] = turns.groupby("call_id").size().reindex(calls["call_id"]).fillna(0).astype(int)
    calls["assigned_char_pct"] = _compute_assigned_char_pct(calls, turns)
    role_unresolved, roster_match = _compute_role_rates(turns, calls)
    calls["role_unresolved_rate"] = role_unresolved
    calls["roster_match_rate"] = roster_match

    if "qa_header_matched" not in calls.columns:
        calls["qa_header_matched"] = calls["qa_text"].fillna("").astype(str).str.len().gt(0)

    write_parquet(calls.drop(columns=["roster_map"], errors="ignore"), artifacts["calls"])

    turns = turns.copy()
    turns["n_tokens"] = turns["turn_text"].fillna("").astype(str).apply(count_tokens)
    lm = compute_lm_features(turns.rename(columns={"turn_text": "clean_transcript"}), text_col="clean_transcript", token_col="n_tokens", external_dir=paths.external_dir, logger=logger)
    for col in lm.columns:
        turns[col] = lm[col].values

    turns["is_ai_kw"] = compute_is_ai_kw(turns["turn_text"])
    turns["ai_score_encoder"] = _maybe_transfer_scores(turns, cfg=feature_cfg, paths=paths, logger=logger)
    turns["ai_score_encoder"] = turns["ai_score_encoder"].fillna(0.0).astype("float64")

    rules = turns.apply(lambda r: apply_ai_rules(r, thr_hi=qa_cfg.ai_thr_hi, thr_lo=qa_cfg.ai_thr_lo), axis=1, result_type="expand")
    turns["is_ai_final"] = rules["is_ai_final"].astype(bool)
    turns["needs_review"] = rules["needs_review"].astype(bool)

    write_parquet(turns, artifacts["turns"])

    call_summary = call_level_introducer(turns)
    write_parquet(call_summary, artifacts["analysis"])

    # Analysis tables
    q_rate = ai_question_rate_by_quarter(turns)
    a_rate = ai_answer_rate_by_quarter(turns)
    vol = ai_questions_by_sector_quarter(turns)
    intro_q = introduced_by_by_quarter(call_summary, calls)
    intro_s = introduced_by_by_sector(call_summary, calls)
    pos = ai_first_turn_position_distribution(call_summary)
    unc_ai = answers_uncertainty_by_ai(turns)
    unc_intro = answers_uncertainty_by_introducer(turns, call_summary)

    write_csv(q_rate, tbl_dir / "ai_question_rate_by_quarter.csv")
    write_csv(a_rate, tbl_dir / "ai_answer_rate_by_quarter.csv")
    write_csv(vol, tbl_dir / "ai_questions_by_sector_quarter.csv")
    write_csv(intro_q, tbl_dir / "who_introduces_ai_first_by_quarter.csv")
    write_csv(intro_s, tbl_dir / "who_introduces_ai_first_by_sector.csv")
    write_csv(pos, tbl_dir / "ai_first_turn_position_distribution.csv")
    write_csv(unc_ai, tbl_dir / "answers_uncertainty_ai_vs_nonai.csv")
    write_csv(unc_intro, tbl_dir / "answers_uncertainty_analystfirst_vs_mgmtfirst.csv")

    validate_tables(calls, turns, qa_pairs, call_summary, dev_mode=run_cfg.dev_mode, qa_cfg=qa_cfg)

    # Figures
    plot_trend_line(
        q_rate,
        x_col="datacqtr",
        y_col="ai_question_rate",
        title="AI Question Rate by Quarter",
        out_path=fig_dir / "trend_ai_question_rate_by_quarter.png",
        logger=logger,
    )
    plot_trend_line(
        a_rate,
        x_col="datacqtr",
        y_col="ai_answer_rate",
        title="AI Answer Rate by Quarter",
        out_path=fig_dir / "trend_ai_answer_rate_by_quarter.png",
        logger=logger,
    )
    plot_stacked_bar(
        intro_q,
        index_col="datacqtr",
        category_col="introduced_by",
        value_col="n_calls",
        title="Who Introduces AI First (by Quarter)",
        out_path=fig_dir / "who_introduces_ai_first_by_quarter.png",
        logger=logger,
    )
    plot_stacked_bar(
        intro_s,
        index_col="sector",
        category_col="introduced_by",
        value_col="n_calls",
        title="Who Introduces AI First (by Sector)",
        out_path=fig_dir / "who_introduces_ai_first_by_sector.png",
        logger=logger,
    )
    plot_bar(
        pos,
        x_col="bucket",
        y_col="n_calls",
        title="AI First Turn Position (Normalized)",
        out_path=fig_dir / "ai_first_turn_position_distribution.png",
        logger=logger,
    )
    plot_bar(
        unc_ai,
        x_col="ai_bucket",
        y_col="uncertainty_mean",
        title="Answer Uncertainty: AI vs Non-AI",
        out_path=fig_dir / "answers_uncertainty_ai_vs_nonai.png",
        logger=logger,
    )
    plot_bar(
        unc_intro,
        x_col="introduced_by",
        y_col="uncertainty_mean",
        title="Answer Uncertainty by Who Introduced AI",
        out_path=fig_dir / "answers_uncertainty_analystfirst_vs_mgmtfirst.png",
        logger=logger,
    )

    # Optional topic plot placeholder + RESULTS note
    optional_fig = fig_dir / "ai_question_subtopics_trend.png"
    results_md = paths.outputs_dir / "RESULTS.md"
    if not optional_fig.exists():
        plot_placeholder(optional_fig, reason="topic modeling not run")
        results_md.parent.mkdir(parents=True, exist_ok=True)
        with results_md.open("a", encoding="utf-8") as f:
            f.write(f\"\\nSKIPPED: ai_question_subtopics_trend.png - topic modeling not run\\n\")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA pipeline")
    parser.add_argument("--dev", action="store_true", help="Run in DEV mode")
    parser.add_argument("--dev-sample-n", type=int, default=800)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    run_cfg = RunConfig(dev_mode=args.dev, dev_sample_n=args.dev_sample_n, recompute=args.recompute)
    run_all_qa(run_cfg, dataset_cfg=DatasetConfig(), feature_cfg=FeatureConfig(), qa_cfg=QAConfig())


if __name__ == "__main__":
    main()
