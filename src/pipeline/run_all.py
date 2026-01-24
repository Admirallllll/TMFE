from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AI_SEED_STATEMENTS, DatasetConfig, FeatureConfig, Paths, RunConfig
from src.data.build_panel import build_panel_with_targets
from src.data.load_hf import load_hf_dataset_to_df
from src.data.preprocess import preprocess_base_table
from src.features.ani import compute_ani_features
from src.features.document_embeddings import compute_document_embeddings
from src.features.embeddings_similarity import compute_embedding_similarity_features
from src.features.sentiment_lm import compute_lm_features
from src.features.tfidf_features import fit_tfidf, top_ngrams
from src.features.topics_bertopic import compute_topic_features
from src.models.evaluation import run_model_benchmarks, run_rolling_benchmarks
from src.utils.io import read_parquet, write_csv, write_parquet
from src.utils.logging import setup_logging
from src.utils.time import datacqtr_to_index
from src.viz.eda_plots import generate_eda_figures
from src.viz.results_plots import plot_model_results, plot_pre_post_sector, plot_sector_heterogeneity


def _artifact_paths(base_dir: Path, *, dev_mode: bool) -> dict[str, Path]:
    if dev_mode:
        base_dir = base_dir / "dev"
    base_dir.mkdir(parents=True, exist_ok=True)

    return {
        "raw": base_dir / "raw_snapshot.parquet",
        "base": base_dir / "base_panel.parquet",
        "panel": base_dir / "panel_with_targets.parquet",
        "ani": base_dir / "features_ani.parquet",
        "lm": base_dir / "features_lm.parquet",
        "topics": base_dir / "features_topics.parquet",
        "doc_emb": base_dir / "document_embeddings.npy",
        "emb": base_dir / "features_embeddings.parquet",
        "model_df": base_dir / "modeling_dataset.parquet",
    }


def _maybe_load(path: Path, *, recompute: bool) -> pd.DataFrame | None:
    if (not recompute) and path.exists():
        return read_parquet(path)
    return None


def _maybe_load_embeddings(path: Path, *, recompute: bool, expected_rows: int) -> np.ndarray | None:
    if recompute or (not path.exists()):
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        return None
    return arr


def _compute_pre_post(df: pd.DataFrame, *, split_datacqtr: str) -> pd.DataFrame:
    split_idx = datacqtr_to_index(split_datacqtr)
    period = np.where(df["quarter_index"].astype(int) >= split_idx, "post", "pre")
    d = df.copy()
    d["period"] = period

    metrics = ["ani_kw_per1k", "ai_topic_share"]
    rows: list[dict[str, object]] = []

    for metric in metrics:
        for p, g in d.groupby("period"):
            rows.append({"group": "overall", "sector": "ALL", "period": p, "metric": metric, "value": float(g[metric].mean()), "n": int(len(g))})
        for (sector, p), g in d.groupby(["sector", "period"]):
            rows.append({"group": "sector", "sector": str(sector), "period": p, "metric": metric, "value": float(g[metric].mean()), "n": int(len(g))})

    return pd.DataFrame(rows)


def _compute_sector_heterogeneity(
    df: pd.DataFrame,
    *,
    train_end_datacqtr: str,
    test_start_datacqtr: str,
    numeric_features_meta: list[str],
    numeric_features_text: list[str],
    categorical_features: list[str],
    seed: int,
    logger,
) -> pd.DataFrame:
    train_end_idx = datacqtr_to_index(train_end_datacqtr)
    test_start_idx = datacqtr_to_index(test_start_datacqtr)

    rows: list[dict[str, object]] = []
    sectors = [s for s in sorted(df["sector"].dropna().unique().tolist())]
    for sector in sectors:
        sub = df.loc[df["sector"] == sector].copy()
        train = sub.loc[sub["quarter_index"].astype(int) <= train_end_idx]
        test = sub.loc[sub["quarter_index"].astype(int) >= test_start_idx]
        if len(train) < 300 or len(test) < 80:
            continue

        from sklearn.metrics import roc_auc_score

        from src.models.train_classification import build_classification_models
        from src.models.train_regression import build_regression_models
        from src.models.evaluation import regression_metrics

        reg_bm2 = build_regression_models(numeric_features=numeric_features_meta, categorical_features=categorical_features, seed=seed)[0].pipeline
        reg_m1 = build_regression_models(numeric_features=numeric_features_text, categorical_features=categorical_features, seed=seed)[1].pipeline
        y_train_reg = train["delta_peforw_qavg"].astype(float).to_numpy()
        y_test_reg = test["delta_peforw_qavg"].astype(float).to_numpy()
        reg_bm2.fit(train, y_train_reg)
        reg_m1.fit(train, y_train_reg)
        bm2_rmse = regression_metrics(y_test_reg, reg_bm2.predict(test))["rmse"]
        m1_rmse = regression_metrics(y_test_reg, reg_m1.predict(test))["rmse"]

        cls_bm2 = build_classification_models(numeric_features=numeric_features_meta, categorical_features=categorical_features, seed=seed)[0].pipeline
        cls_m1 = build_classification_models(numeric_features=numeric_features_text, categorical_features=categorical_features, seed=seed)[1].pipeline
        y_train_cls = train["valuation_upgrade"].astype(int).to_numpy()
        y_test_cls = test["valuation_upgrade"].astype(int).to_numpy()
        cls_bm2.fit(train, y_train_cls)
        cls_m1.fit(train, y_train_cls)
        bm2_auc = float(roc_auc_score(y_test_cls, cls_bm2.predict_proba(test)[:, 1]))
        m1_auc = float(roc_auc_score(y_test_cls, cls_m1.predict_proba(test)[:, 1]))

        rows.append({"sector": str(sector), "metric": "rmse", "bm2": bm2_rmse, "m1": m1_rmse, "delta": bm2_rmse - m1_rmse, "n_train": int(len(train)), "n_test": int(len(test))})
        rows.append({"sector": str(sector), "metric": "auc", "bm2": bm2_auc, "m1": m1_auc, "delta": m1_auc - bm2_auc, "n_train": int(len(train)), "n_test": int(len(test))})

    return pd.DataFrame(rows)


def run_all(run_cfg: RunConfig, *, dataset_cfg: DatasetConfig, feature_cfg: FeatureConfig) -> None:
    paths = Paths()
    artifacts = _artifact_paths(paths.processed_dir, dev_mode=run_cfg.dev_mode)

    fig_dir = paths.figures_dir / ("dev" if run_cfg.dev_mode else "")
    tbl_dir = paths.tables_dir / ("dev" if run_cfg.dev_mode else "")
    model_dir = paths.models_dir / ("dev" if run_cfg.dev_mode else "")
    log_path = paths.logs_dir / ("pipeline_dev.log" if run_cfg.dev_mode else "pipeline.log")
    logger = setup_logging(log_path)

    for d in (fig_dir, tbl_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    start = time.time()
    logger.info(f"Run config: dev_mode={run_cfg.dev_mode} dev_sample_n={run_cfg.dev_sample_n} recompute={run_cfg.recompute}")

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
        )
        write_parquet(raw, artifacts["raw"])
        logger.info(f"Saved raw snapshot: {artifacts['raw']}")

    base = _maybe_load(artifacts["base"], recompute=run_cfg.recompute)
    if base is None:
        res = preprocess_base_table(raw, remove_boilerplate=run_cfg.remove_boilerplate, logger=logger)
        base = res.df
        write_parquet(base, artifacts["base"])
        logger.info(f"Saved base panel: {artifacts['base']}")

    panel = _maybe_load(artifacts["panel"], recompute=run_cfg.recompute)
    if panel is None:
        panel = build_panel_with_targets(base, logger=logger)
        write_parquet(panel, artifacts["panel"])
        logger.info(f"Saved panel with targets: {artifacts['panel']}")

    ani = _maybe_load(artifacts["ani"], recompute=run_cfg.recompute)
    if ani is None:
        ani = compute_ani_features(panel, text_col="clean_transcript", token_col="n_tokens", logger=logger)
        write_parquet(ani, artifacts["ani"])

    lm = _maybe_load(artifacts["lm"], recompute=run_cfg.recompute)
    if lm is None:
        lm = compute_lm_features(panel, text_col="clean_transcript", token_col="n_tokens", external_dir=paths.external_dir, logger=logger)
        write_parquet(lm, artifacts["lm"])

    topics = _maybe_load(artifacts["topics"], recompute=run_cfg.recompute)
    emb = _maybe_load(artifacts["emb"], recompute=run_cfg.recompute)

    doc_emb: np.ndarray | None = None
    if (topics is None) or (emb is None):
        doc_emb = _maybe_load_embeddings(artifacts["doc_emb"], recompute=run_cfg.recompute, expected_rows=len(panel))
        if doc_emb is None:
            try:
                doc_emb_res = compute_document_embeddings(
                    panel,
                    text_col="clean_transcript",
                    model_name=feature_cfg.embeddings_model,
                    max_chars_per_chunk=feature_cfg.embeddings_max_chars_per_chunk,
                    max_chunks_per_doc=feature_cfg.embeddings_max_chunks_per_doc,
                    batch_size=feature_cfg.embeddings_batch_size,
                    device=feature_cfg.embeddings_device,
                    logger=logger,
                )
                doc_emb = doc_emb_res.embeddings
                artifacts["doc_emb"].parent.mkdir(parents=True, exist_ok=True)
                np.save(artifacts["doc_emb"], doc_emb.astype("float32", copy=False))
                logger.info(f"Saved document embeddings: {artifacts['doc_emb']}")
            except Exception as e:
                logger.info(f"Document embeddings failed ({e}); continuing without cached embeddings")
                doc_emb = None

    if topics is None:
        panel_for_topics = panel
        if "ani_kw_per1k" not in panel.columns and {"ani_any", "ani_kw_per1k"}.issubset(set(ani.columns)):
            panel_for_topics = panel.assign(
                ani_any=ani["ani_any"].to_numpy(),
                ani_kw_per1k=ani["ani_kw_per1k"].to_numpy(),
            )
        topic_res = compute_topic_features(
            panel_for_topics,
            text_col="clean_transcript",
            model_dir=model_dir,
            method=feature_cfg.topic_model,
            doc_embeddings=doc_emb,
            bertopic_calculate_probabilities=feature_cfg.bertopic_calculate_probabilities,
            lda_num_topics=feature_cfg.lda_num_topics,
            lda_passes=feature_cfg.lda_passes,
            lda_chunksize=feature_cfg.lda_chunksize,
            logger=logger,
        )
        topics = topic_res.features
        write_parquet(topics, artifacts["topics"])
        logger.info(f"Topics method used: {topic_res.method}, ai_topics={len(topic_res.ai_topic_ids)}")

    if emb is None:
        emb_res = compute_embedding_similarity_features(
            panel,
            text_col="clean_transcript",
            seed_statements=AI_SEED_STATEMENTS,
            model_name=feature_cfg.embeddings_model,
            max_chars_per_chunk=feature_cfg.embeddings_max_chars_per_chunk,
            max_chunks_per_doc=feature_cfg.embeddings_max_chunks_per_doc,
            batch_size=feature_cfg.embeddings_batch_size,
            device=feature_cfg.embeddings_device,
            model_dir=model_dir,
            logger=logger,
            doc_embeddings=doc_emb,
        )
        emb = emb_res.features
        write_parquet(emb, artifacts["emb"])
        logger.info(f"Embedding similarity method used: {emb_res.method}")

    model_df = _maybe_load(artifacts["model_df"], recompute=run_cfg.recompute)
    if model_df is None:
        model_df = pd.concat([panel.reset_index(drop=True), ani, lm, topics, emb], axis=1)
        write_parquet(model_df, artifacts["model_df"])
        logger.info(f"Saved modeling dataset: {artifacts['model_df']}")

    tfidf_meta_path = model_dir / "tfidf_top_ngrams.csv"
    if run_cfg.recompute or (not (model_dir / "tfidf_vectorizer.joblib").exists()):
        tfidf_art = fit_tfidf(
            model_df,
            text_col="clean_transcript",
            model_dir=model_dir,
            min_df=feature_cfg.tfidf_min_df,
            max_df=feature_cfg.tfidf_max_df,
            ngram_range=feature_cfg.tfidf_ngram_range,
            max_features=feature_cfg.tfidf_max_features,
            logger=logger,
        )
        top = top_ngrams(tfidf_art.vectorizer_path, tfidf_art.matrix_path, top_k=40)
        write_csv(top, tfidf_meta_path)
        logger.info(f"Saved TF-IDF top ngrams: {tfidf_meta_path}")

    generate_eda_figures(model_df, figures_dir=fig_dir, logger=logger)

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

    model_dir.mkdir(parents=True, exist_ok=True)
    results = run_model_benchmarks(
        model_df,
        target_reg="delta_peforw_qavg",
        target_cls="valuation_upgrade",
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text,
        categorical_features=categorical,
        train_end_datacqtr=run_cfg.train_end_datacqtr,
        test_start_datacqtr=run_cfg.test_start_datacqtr,
        seed=run_cfg.seed,
        model_dir=model_dir,
        logger=logger,
    )
    write_csv(results, tbl_dir / "model_results.csv")
    plot_model_results(results, fig_dir / "results_model_benchmarks.png")

    rolling = run_rolling_benchmarks(
        model_df,
        target_reg="delta_peforw_qavg",
        target_cls="valuation_upgrade",
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text,
        categorical_features=categorical,
        seed=run_cfg.seed,
        logger=logger,
    )
    if not rolling.empty:
        write_csv(rolling, tbl_dir / "rolling_results.csv")

    robust = run_model_benchmarks(
        model_df,
        target_reg="delta_peforw_eoq",
        target_cls="valuation_upgrade",
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text,
        categorical_features=categorical,
        train_end_datacqtr=run_cfg.train_end_datacqtr,
        test_start_datacqtr=run_cfg.test_start_datacqtr,
        seed=run_cfg.seed,
        model_dir=model_dir,
        logger=logger,
    )
    robust_reg = robust.loc[robust["task"] == "regression"].reset_index(drop=True)
    write_csv(robust_reg, tbl_dir / "model_results_regression_robustness_eoq.csv")

    pre_post = _compute_pre_post(model_df, split_datacqtr="2022Q4")
    write_csv(pre_post, tbl_dir / "pre_post_2022Q4_summary.csv")
    plot_pre_post_sector(pre_post, fig_dir / "results_pre_post_ani_by_sector.png", metric_col="ani_kw_per1k")
    plot_pre_post_sector(pre_post, fig_dir / "results_pre_post_ai_topic_share_by_sector.png", metric_col="ai_topic_share")

    sector_table = _compute_sector_heterogeneity(
        model_df,
        train_end_datacqtr=run_cfg.train_end_datacqtr,
        test_start_datacqtr=run_cfg.test_start_datacqtr,
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text,
        categorical_features=["datacqtr", "ticker"],
        seed=run_cfg.seed,
        logger=logger,
    )
    if not sector_table.empty:
        write_csv(sector_table, tbl_dir / "sector_heterogeneity.csv")
        plot_sector_heterogeneity(sector_table, fig_dir / "results_sector_heterogeneity_rmse.png", metric_col="rmse")
        plot_sector_heterogeneity(sector_table, fig_dir / "results_sector_heterogeneity_auc.png", metric_col="auc")

    elapsed = time.time() - start
    logger.info(f"Pipeline complete in {elapsed/60.0:.1f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full AI narratives research pipeline end-to-end.")
    parser.add_argument("--dev", action="store_true", help="Run in DEV_MODE with a smaller sample and dev outputs.")
    parser.add_argument("--dev-sample-n", type=int, default=RunConfig().dev_sample_n, help="Number of documents to sample in DEV_MODE.")
    parser.add_argument("--recompute", action="store_true", help="Ignore caches and recompute all artifacts.")
    parser.add_argument(
        "--device",
        type=str,
        default=FeatureConfig().embeddings_device,
        help="Embedding device for sentence-transformers (cuda/cpu/auto). Default: cuda.",
    )
    parser.add_argument(
        "--emb-batch-size",
        type=int,
        default=FeatureConfig().embeddings_batch_size,
        help="Batch size for sentence-transformers encoding. Increase on GPUs if you have VRAM.",
    )
    parser.add_argument(
        "--emb-max-chunks",
        type=int,
        default=int(FeatureConfig().embeddings_max_chunks_per_doc or 0),
        help="Max chunks per transcript for embeddings (0 = no cap). Lower is faster.",
    )
    parser.add_argument(
        "--no-topic-probs",
        action="store_true",
        help="Disable BERTopic probability calculation (faster; ai_topic_share becomes a 0/1 indicator).",
    )
    args = parser.parse_args()

    run_cfg = RunConfig(dev_mode=bool(args.dev), dev_sample_n=int(args.dev_sample_n), recompute=bool(args.recompute))
    emb_max_chunks = int(args.emb_max_chunks)
    feature_cfg = FeatureConfig(
        embeddings_device=str(args.device),
        embeddings_batch_size=int(args.emb_batch_size),
        embeddings_max_chunks_per_doc=(None if emb_max_chunks <= 0 else emb_max_chunks),
        bertopic_calculate_probabilities=(not bool(args.no_topic_probs)),
    )
    run_all(run_cfg, dataset_cfg=DatasetConfig(), feature_cfg=feature_cfg)


if __name__ == "__main__":
    main()
