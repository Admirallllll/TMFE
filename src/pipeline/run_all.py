from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AI_SEED_STATEMENTS, DatasetConfig, FeatureColumns, FeatureConfig, Paths, RunConfig
from src.data.build_panel import build_panel_with_targets
from src.data.load_hf import load_hf_dataset_to_df
from src.data.load_kaggle_ai_media import load_ai_media_dataset
from src.data.preprocess import preprocess_base_table
from src.features.ani import compute_ani_features
from src.features.document_embeddings import compute_document_embeddings
from src.features.embeddings_similarity import compute_embedding_similarity_features
from src.features.sentiment_lm import compute_lm_features
from src.features.tfidf_features import fit_tfidf, top_ngrams
from src.features.transfer_inference import compute_transfer_features
from src.features.transfer_tag_labeling import build_transfer_labels
from src.features.topics_bertopic import compute_topic_features
from src.models.evaluation import build_transfer_ablation_table, run_model_benchmarks, run_rolling_benchmarks
from src.models.train_transfer_encoder import train_or_load_transfer_encoder
from src.utils.io import read_parquet, write_csv, write_parquet
from src.utils.logging import setup_logging
from src.utils.time import datacqtr_to_index
from src.viz.eda_plots import generate_eda_figures
from src.viz.results_from_tables import generate_results_figures, generate_transfer_ablation_figures
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
        "transfer": base_dir / "features_transfer.parquet",
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


def _evaluate_single_sector(
    sector: str,
    df: pd.DataFrame,
    train_end_idx: int,
    test_start_idx: int,
    reg_bm2_template,
    reg_m1_template,
    cls_bm2_template,
    cls_m1_template,
) -> list[dict[str, object]]:
    """Evaluate models for a single sector. Used for parallel processing."""
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    from src.models.evaluation import regression_metrics

    sub = df.loc[df["sector"] == sector].copy()
    train = sub.loc[sub["quarter_index"].astype(int) <= train_end_idx]
    test = sub.loc[sub["quarter_index"].astype(int) >= test_start_idx]

    if len(train) < 300 or len(test) < 80:
        return []

    # Clone model templates instead of rebuilding
    reg_bm2 = clone(reg_bm2_template)
    reg_m1 = clone(reg_m1_template)
    cls_bm2 = clone(cls_bm2_template)
    cls_m1 = clone(cls_m1_template)

    y_train_reg = train["delta_peforw_qavg"].astype(float).to_numpy()
    y_test_reg = test["delta_peforw_qavg"].astype(float).to_numpy()
    reg_bm2.fit(train, y_train_reg)
    reg_m1.fit(train, y_train_reg)
    bm2_rmse = regression_metrics(y_test_reg, reg_bm2.predict(test))["rmse"]
    m1_rmse = regression_metrics(y_test_reg, reg_m1.predict(test))["rmse"]

    y_train_cls = train["valuation_upgrade"].astype(int).to_numpy()
    y_test_cls = test["valuation_upgrade"].astype(int).to_numpy()
    cls_bm2.fit(train, y_train_cls)
    cls_m1.fit(train, y_train_cls)
    bm2_auc = float(roc_auc_score(y_test_cls, cls_bm2.predict_proba(test)[:, 1]))
    m1_auc = float(roc_auc_score(y_test_cls, cls_m1.predict_proba(test)[:, 1]))

    return [
        {"sector": str(sector), "metric": "rmse", "bm2": bm2_rmse, "m1": m1_rmse, "delta": bm2_rmse - m1_rmse, "n_train": int(len(train)), "n_test": int(len(test))},
        {"sector": str(sector), "metric": "auc", "bm2": bm2_auc, "m1": m1_auc, "delta": m1_auc - bm2_auc, "n_train": int(len(train)), "n_test": int(len(test))},
    ]


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
    from joblib import Parallel, delayed

    from src.models.train_classification import build_classification_models
    from src.models.train_regression import build_regression_models

    train_end_idx = datacqtr_to_index(train_end_datacqtr)
    test_start_idx = datacqtr_to_index(test_start_datacqtr)

    # Build model templates once, outside the loop
    reg_bm2_template = build_regression_models(
        numeric_features=numeric_features_meta, categorical_features=categorical_features, seed=seed
    )[0].pipeline
    reg_m1_template = build_regression_models(
        numeric_features=numeric_features_text, categorical_features=categorical_features, seed=seed
    )[1].pipeline
    cls_bm2_template = build_classification_models(
        numeric_features=numeric_features_meta, categorical_features=categorical_features, seed=seed
    )[0].pipeline
    cls_m1_template = build_classification_models(
        numeric_features=numeric_features_text, categorical_features=categorical_features, seed=seed
    )[1].pipeline

    sectors = [s for s in sorted(df["sector"].dropna().unique().tolist())]
    logger.info(f"Running sector heterogeneity analysis on {len(sectors)} sectors (parallel)")

    # Parallel processing across sectors
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_evaluate_single_sector)(
            sector, df, train_end_idx, test_start_idx,
            reg_bm2_template, reg_m1_template, cls_bm2_template, cls_m1_template
        )
        for sector in sectors
    )

    # Flatten results
    rows = [row for sector_rows in results for row in sector_rows]
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
            start_year=dataset_cfg.start_year,
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

    transfer = _maybe_load(artifacts["transfer"], recompute=run_cfg.recompute)
    transfer_raw_path = paths.transfer_data_dir / "ai_media_raw.parquet"
    transfer_labeled_path = paths.transfer_data_dir / "ai_media_labeled.parquet"
    transfer_audit_path = paths.transfer_table_dir / "tag_audit.csv"
    if transfer is None:
        paths.transfer_data_dir.mkdir(parents=True, exist_ok=True)
        paths.transfer_table_dir.mkdir(parents=True, exist_ok=True)

        if (not run_cfg.recompute) and transfer_raw_path.exists():
            ai_media_raw = read_parquet(transfer_raw_path)
            logger.info(f"Loaded cached AI media raw data: {transfer_raw_path}")
        else:
            ai_media_raw = load_ai_media_dataset(logger=logger)
            write_parquet(ai_media_raw, transfer_raw_path)
            logger.info(f"Saved AI media raw data: {transfer_raw_path}")

        if (not run_cfg.recompute) and transfer_labeled_path.exists():
            ai_media_labeled = read_parquet(transfer_labeled_path)
            logger.info(f"Loaded cached AI media labeled data: {transfer_labeled_path}")
        else:
            ai_media_labeled, tag_audit = build_transfer_labels(
                ai_media_raw,
                min_tag_freq=feature_cfg.transfer_min_tag_freq,
                label_margin=feature_cfg.transfer_label_margin,
                logger=logger,
            )
            write_parquet(ai_media_labeled, transfer_labeled_path)
            write_csv(tag_audit, transfer_audit_path)
            logger.info(f"Saved AI media labeled data: {transfer_labeled_path}")
            logger.info(f"Saved transfer tag audit: {transfer_audit_path}")

        transfer_bundle = train_or_load_transfer_encoder(
            ai_media_labeled,
            cfg=feature_cfg,
            paths=paths,
            logger=logger,
            force_retrain=run_cfg.transfer_retrain,
            max_train_samples=run_cfg.transfer_max_train_samples,
            seed=run_cfg.seed,
        )
        transfer_res = compute_transfer_features(panel, encoder_bundle=transfer_bundle, cfg=feature_cfg, logger=logger)
        transfer = transfer_res.features
        write_parquet(transfer, artifacts["transfer"])
        logger.info(
            f"Saved transfer features: {artifacts['transfer']} "
            + f"(model={transfer_res.model_name}, threshold={transfer_res.threshold:.2f})"
        )

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
            except RuntimeError as e:
                logger.error(f"Document embeddings failed (RuntimeError): {e}")
                doc_emb = None
            except Exception as e:
                logger.warning(f"Document embeddings failed (unexpected error): {e}")
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
        model_df = pd.concat([panel.reset_index(drop=True), ani, lm, topics, emb, transfer], axis=1)
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

    # Use centralized feature column registry
    feat_cols = FeatureColumns()
    numeric_meta = list(feat_cols.meta_numeric)
    transfer_cols = [c for c in feat_cols.text_numeric if c.startswith("transfer_")]
    text_only_cols = [c for c in feat_cols.text_numeric if c not in set(transfer_cols)]
    numeric_text_base = numeric_meta + text_only_cols
    numeric_text_transfer = numeric_meta + list(feat_cols.text_numeric)
    categorical = list(feat_cols.categorical)

    model_dir.mkdir(parents=True, exist_ok=True)
    baseline_results = run_model_benchmarks(
        model_df,
        target_reg="delta_peforw_qavg",
        target_cls="valuation_upgrade",
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text_base,
        categorical_features=categorical,
        train_end_datacqtr=run_cfg.train_end_datacqtr,
        test_start_datacqtr=run_cfg.test_start_datacqtr,
        seed=run_cfg.seed,
        model_dir=model_dir,
        logger=logger,
    )

    results = run_model_benchmarks(
        model_df,
        target_reg="delta_peforw_qavg",
        target_cls="valuation_upgrade",
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text_transfer,
        categorical_features=categorical,
        train_end_datacqtr=run_cfg.train_end_datacqtr,
        test_start_datacqtr=run_cfg.test_start_datacqtr,
        seed=run_cfg.seed,
        model_dir=model_dir,
        logger=logger,
    )
    write_csv(results, tbl_dir / "model_results.csv")
    plot_model_results(results, fig_dir / "results_model_benchmarks.png")

    # Generate additional model results plots
    generate_results_figures(tbl_dir / "model_results.csv", fig_dir, logger=logger)

    ablation = build_transfer_ablation_table(baseline_results, results)
    if not ablation.empty:
        write_csv(ablation, tbl_dir / "model_results_transfer_ablation.csv")
        generate_transfer_ablation_figures(tbl_dir / "model_results_transfer_ablation.csv", fig_dir, logger=logger)

    rolling = run_rolling_benchmarks(
        model_df,
        target_reg="delta_peforw_qavg",
        target_cls="valuation_upgrade",
        numeric_features_meta=numeric_meta,
        numeric_features_text=numeric_text_transfer,
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
        numeric_features_text=numeric_text_transfer,
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
        numeric_features_text=numeric_text_transfer,
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
    parser.add_argument(
        "--start-year",
        type=int,
        default=DatasetConfig().start_year,
        help="Filter data to include only documents from this year onwards (default: 2018). Set to 0 to disable.",
    )
    parser.add_argument(
        "--transfer-device",
        type=str,
        default=FeatureConfig().transfer_device,
        help="Device for transfer encoder training/inference (auto/cuda/cpu/mps).",
    )
    parser.add_argument(
        "--transfer-epochs",
        type=int,
        default=FeatureConfig().transfer_epochs,
        help="Training epochs for transfer encoder.",
    )
    parser.add_argument(
        "--transfer-batch-size",
        type=int,
        default=FeatureConfig().transfer_batch_size,
        help="Batch size for transfer encoder training and inference.",
    )
    parser.add_argument(
        "--transfer-lr",
        type=float,
        default=FeatureConfig().transfer_lr,
        help="Learning rate for transfer encoder.",
    )
    parser.add_argument(
        "--transfer-retrain",
        action="store_true",
        help="Force retraining transfer encoder even if cached model exists.",
    )
    parser.add_argument(
        "--transfer-max-train-samples",
        type=int,
        default=0,
        help="Optional cap on labeled source samples for transfer training (0 = no cap).",
    )
    args = parser.parse_args()

    run_cfg = RunConfig(
        dev_mode=bool(args.dev),
        dev_sample_n=int(args.dev_sample_n),
        recompute=bool(args.recompute),
        transfer_retrain=bool(args.transfer_retrain),
        transfer_max_train_samples=(None if int(args.transfer_max_train_samples) <= 0 else int(args.transfer_max_train_samples)),
    )
    emb_max_chunks = int(args.emb_max_chunks)
    feature_cfg = FeatureConfig(
        embeddings_device=str(args.device),
        embeddings_batch_size=int(args.emb_batch_size),
        embeddings_max_chunks_per_doc=(None if emb_max_chunks <= 0 else emb_max_chunks),
        bertopic_calculate_probabilities=(not bool(args.no_topic_probs)),
        transfer_device=str(args.transfer_device),
        transfer_epochs=int(args.transfer_epochs),
        transfer_batch_size=int(args.transfer_batch_size),
        transfer_lr=float(args.transfer_lr),
    )
    start_year = int(args.start_year) if args.start_year and int(args.start_year) > 0 else None
    dataset_cfg = DatasetConfig(start_year=start_year)
    run_all(run_cfg, dataset_cfg=dataset_cfg, feature_cfg=feature_cfg)


if __name__ == "__main__":
    main()
