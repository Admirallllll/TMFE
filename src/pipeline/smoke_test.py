from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AI_SEED_STATEMENTS, FeatureConfig, Paths, RunConfig
from src.data.build_panel import build_panel_with_targets
from src.data.preprocess import preprocess_base_table
from src.features.ani import compute_ani_features
from src.features.embeddings_similarity import compute_embedding_similarity_features
from src.features.sentiment_lm import compute_lm_features
from src.features.topics_bertopic import compute_topic_features
from src.models.evaluation import run_model_benchmarks
from src.utils.logging import setup_logging


def _make_synthetic_df(n_tickers: int, years: list[int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    sectors = ["Tech", "Health", "Finance", "Industrials"]
    quarters = [1, 2, 3, 4]

    rows = []
    for y in years:
        for q in quarters:
            datacqtr = f"{y}Q{q}"
            for t in tickers:
                transcript = "We discussed revenue and margins."
                if rng.random() < 0.15:
                    transcript += " We are investing in AI and machine learning."
                if rng.random() < 0.08:
                    transcript += " Generative AI and LLM capabilities are a priority."

                pe = float(rng.normal(18.0, 4.0))
                eps_tr = float(rng.normal(2.0, 0.5))
                rows.append(
                    {
                        "ticker": t,
                        "company": f"Company {t}",
                        "cik": int(rng.integers(1_000_000, 9_999_999)),
                        "sector": rng.choice(sectors),
                        "industry": "Synthetic",
                        "earnings_date": f"{y}-{q*3:02d}-15",
                        "datacqtr": datacqtr,
                        "datafqtr": datacqtr,
                        "year": float(y),
                        "quarter": float(q),
                        "eps12mtrailing_qavg": eps_tr,
                        "eps12mtrailing_eoq": eps_tr,
                        "eps12mfwd_qavg": eps_tr + float(rng.normal(0.1, 0.1)),
                        "eps12mfwd_eoq": eps_tr + float(rng.normal(0.1, 0.1)),
                        "eps_lt": float(rng.normal(0.05, 0.02)),
                        "peforw_qavg": pe,
                        "peforw_eoq": pe + float(rng.normal(0.0, 0.5)),
                        "transcript": transcript,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the AI narratives pipeline (synthetic data).")
    parser.add_argument("--rows", type=int, default=400, help="Approximate number of rows to generate.")
    args = parser.parse_args()

    run_cfg = RunConfig(dev_mode=True, dev_sample_n=int(args.rows), recompute=True)
    paths = Paths()
    logger = setup_logging(paths.logs_dir / "smoke_test.log")

    years = [2021, 2022, 2023]
    n_tickers = max(5, int(run_cfg.dev_sample_n / (len(years) * 4)))
    raw = _make_synthetic_df(n_tickers=n_tickers, years=years, seed=run_cfg.seed)

    base = preprocess_base_table(raw, remove_boilerplate=False, logger=logger).df
    panel = build_panel_with_targets(base, logger=logger)

    ani = compute_ani_features(panel, logger=logger)
    lm = compute_lm_features(panel, external_dir=paths.external_dir, logger=logger)
    topics = compute_topic_features(
        panel,
        model_dir=paths.models_dir / "smoke",
        method="auto",
        lda_num_topics=10,
        lda_passes=2,
        lda_chunksize=500,
        logger=logger,
    ).features
    emb = compute_embedding_similarity_features(
        panel,
        text_col="clean_transcript",
        seed_statements=AI_SEED_STATEMENTS,
        model_name=FeatureConfig().embeddings_model,
        max_chars_per_chunk=FeatureConfig().embeddings_max_chars_per_chunk,
        model_dir=paths.models_dir / "smoke",
        logger=logger,
    ).features

    df = pd.concat([panel.reset_index(drop=True), ani, lm, topics, emb], axis=1)

    numeric_meta = [
        "n_tokens",
        "n_chars",
        "lag_peforw_qavg",
        "lag_eps12mtrailing_qavg",
        "lag_eps12mtrailing_eoq",
        "lag_eps12mfwd_qavg",
        "lag_eps12mfwd_eoq",
        "lag_eps_lt",
    ]
    numeric_text = numeric_meta + [
        "ani_kw_per1k",
        "ani_ai_core_per1k",
        "ani_ml_per1k",
        "ani_llm_per1k",
        "ani_genai_per1k",
        "lm_pos_per1k",
        "lm_neg_per1k",
        "lm_unc_per1k",
        "lm_net_tone_per1k",
        "ai_topic_share",
        "ai_sim_mean",
        "ai_sim_max",
    ]
    categorical = ["sector", "datacqtr", "ticker"]

    results = run_model_benchmarks(
        df,
        target_reg="delta_peforw_qavg",
        target_cls="valuation_upgrade",
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text,
        categorical_features=categorical,
        train_end_datacqtr=run_cfg.train_end_datacqtr,
        test_start_datacqtr=run_cfg.test_start_datacqtr,
        seed=run_cfg.seed,
        model_dir=paths.models_dir / "smoke",
        logger=logger,
    )

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
