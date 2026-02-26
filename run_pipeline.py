"""
Main Pipeline Script

End-to-end orchestration of the S&P 500 AI Narrative Text Mining project.

Stages (ai_method dependent):
1. Parse transcripts (Speech/Q&A splitting)
2. Split into sentences
3. Compute keyword baseline
4. Topic modeling per quarter (topic only)
5. Compute AI intensity metrics + visualizations
6. Compute initiation scores + visualizations
7. Run analyses (time series, quadrants, regression)
8. Additional visualizations (company rankings, wordclouds)
"""

import os
import argparse
from datetime import datetime
import json
import hashlib
import platform
import subprocess
import sys
import traceback


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        for stream in self.streams:
            if hasattr(stream, "isatty") and stream.isatty():
                return True
        return False

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", None)


def run_pipeline(
    input_dataset: str = "final_dataset.parquet",
    wrds_path: str = "Sp500_meta_data.csv",
    output_dir: str = "outputs",
    dev_mode: bool = False,
    dev_sample: int = 100,
    seed: int = 42,
    ai_method: str = "topic",
    kw_workers: int | None = None,
    metrics_workers: int | None = None,
    run_lasso: bool = True,
    lasso_max_features: int = 5000,
    lasso_ngram_max: int = 2,
    lasso_cv: int = 5,
    lasso_skip_cv_pred: bool = False,
):
    """
    Run the full pipeline.
    
    Args:
        input_dataset: Path to earnings call dataset
        wrds_path: Path to WRDS metadata
        output_dir: Output directory
        dev_mode: Development mode (small samples)
        dev_sample: Sample size for dev mode
        seed: Random seed for reproducibility
        ai_method: "kw" (dictionary) or "topic" (topic modeling)
        kw_workers: Number of workers for keyword detection (None = auto)
        metrics_workers: Number of workers for AI intensity metrics (None = auto)
    """
    def sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_git_head() -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None

    # Set seeds (best-effort)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    start_time = datetime.now()

    # Set up logging (capture terminal output)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"pipeline_{start_time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)

    try:
        print("="*70)
        print("S&P 500 AI Narrative Text Mining Pipeline")
        print(f"Started at: {start_time}")
        print(f"Logging to: {log_path}")
        print("="*70)

        # Create output directories
        features_dir = f"{output_dir}/features"
        figures_dir = f"{output_dir}/figures"

        for d in [features_dir, figures_dir]:
            os.makedirs(d, exist_ok=True)

        sample_n = dev_sample if dev_mode else None

        # Validate input dataset schema early (high-stakes fail-fast)
        import pandas as pd
        required_cols = {"ticker", "date", "year", "quarter", "structured_content"}
        try:
            # Read a minimal sample to get actual column names (works with all types)
            sample_df = pd.read_parquet(input_dataset, columns=None)
            ds_cols = set(sample_df.columns.tolist())
            del sample_df
        except Exception as e:
            raise RuntimeError(f"Cannot read input dataset: {e}")
        missing = sorted(required_cols - ds_cols)
        if missing:
            print(f"Available columns: {sorted(ds_cols)}")
            raise RuntimeError(f"Input dataset missing required columns: {missing}")

        key_df = pd.read_parquet(input_dataset, columns=["ticker", "year", "quarter", "date"])
        if key_df[["ticker", "year", "quarter"]].isna().any().any():
            raise RuntimeError("Input dataset has missing values in (ticker, year, quarter). Refusing to proceed.")
        dup_keys = key_df[["ticker", "year", "quarter"]].duplicated().sum()
        if int(dup_keys) != 0:
            raise RuntimeError(f"Input dataset has duplicate (ticker, year, quarter) keys: {int(dup_keys)} duplicate rows")

        # =========================================================================
        # Stage 1: Parse Transcripts
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 1: Parse Transcripts (Speech/Q&A Splitting)")
        print("="*70)

        from src.preprocessing.transcript_parser import process_dataset as parse_transcripts

        parsed_path = f"{features_dir}/parsed_transcripts.parquet"
        if not os.path.exists(parsed_path) or dev_mode:
            parse_transcripts(input_dataset, parsed_path, sample_n)
        else:
            print(f"Using existing parsed transcripts: {parsed_path}")

        # =========================================================================
        # Stage 2: Split into Sentences
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 2: Split into Sentences")
        print("="*70)

        from src.preprocessing.sentence_splitter import create_sentence_dataset

        sentences_path = f"{features_dir}/sentences.parquet"
        if not os.path.exists(sentences_path) or dev_mode:
            create_sentence_dataset(parsed_path, sentences_path, sample_n)
        else:
            print(f"Using existing sentences: {sentences_path}")

        # =========================================================================
        # Stage 3: Keyword Detection (Baseline)
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 3: Keyword Detection (Baseline)")
        print("="*70)

        from src.baselines.keyword_detector import compute_keyword_metrics

        sentences_df = pd.read_parquet(sentences_path)
        sentences_with_kw = compute_keyword_metrics(
            sentences_df,
            num_workers=kw_workers
        )
        sentences_with_kw.to_parquet(f"{features_dir}/sentences_with_keywords.parquet", index=False)

        ai_method = str(ai_method).lower()
        if ai_method not in {"kw", "topic"}:
            raise ValueError("ai_method must be one of: kw, topic")

        # =========================================================================
        # Stage 4: Topic Modeling (Quarterly, LDA) - topic only
        # =========================================================================
        if ai_method == "topic":
            print("\n" + "="*70)
            print("STAGE 4: Topic Modeling (Quarterly, LDA)")
            print("="*70)

            from src.analysis.topic_modeling import run_quarterly_topic_modeling

            run_quarterly_topic_modeling(
                f"{features_dir}/sentences_with_keywords.parquet",
                output_dir=features_dir,
                start_year=2020,
                end_year=2025,
                n_topics=20,
                top_n_words=12,
                filter_ai=True
            )

        # =========================================================================
        # Stage 5: Compute AI Intensity Metrics
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 5: Compute AI Intensity Metrics")
        print("="*70)

        from src.metrics.ai_intensity import compute_all_metrics

        sentences_for_metrics = pd.read_parquet(f"{features_dir}/sentences_with_keywords.parquet")
        compute_all_metrics(sentences_for_metrics, features_dir, figures_dir, num_workers=metrics_workers)

        # =========================================================================
        # Stage 6: Compute Initiation Scores
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 6: Compute AI Initiation Scores")
        print("="*70)

        from src.metrics.initiation_score import compute_all_initiation_metrics

        compute_all_initiation_metrics(sentences_for_metrics, features_dir, figures_dir)

        # =========================================================================
        # Stage 7: Analysis - Time Series
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 7: Time Series Analysis")
        print("="*70)

        from src.analysis.time_series import run_time_series_analysis

        run_time_series_analysis(
            f"{features_dir}/document_metrics.parquet",
            input_dataset,
            figures_dir
        )

        # =========================================================================
        # Stage 8: Analysis - Company Quadrants
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 8: Company Quadrant Analysis")
        print("="*70)

        from src.analysis.company_quadrants import run_quadrant_analysis

        run_quadrant_analysis(
            f"{features_dir}/document_metrics.parquet",
            figures_dir
        )

        # =========================================================================
        # Stage 9: Regression Analysis
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 9: Regression Analysis")
        print("="*70)

        from src.analysis.regression import run_regression_analysis

        run_regression_analysis(
            f"{features_dir}/initiation_scores.parquet",
            f"{features_dir}/document_metrics.parquet",
            wrds_path,
            figures_dir
        )

        # =========================================================================
        # Stage 10: Lasso Text Feature Analysis (Volcano + Coefficients)
        # =========================================================================
        if run_lasso:
            print("\n" + "="*70)
            print("STAGE 10: Lasso Text Feature Analysis")
            print("="*70)

            from src.analysis.lasso_text_features import run_lasso_text_analysis

            lasso_out_dir = os.path.join(figures_dir, "lasso")
            run_lasso_text_analysis(
                sentences_path=f"{features_dir}/sentences_with_keywords.parquet",
                doc_metrics_path=f"{features_dir}/document_metrics.parquet",
                initiation_scores_path=f"{features_dir}/initiation_scores.parquet",
                output_dir=lasso_out_dir,
                max_features=lasso_max_features,
                ngram_range=(1, lasso_ngram_max),
                cv=lasso_cv,
                compute_cv_predictions=not lasso_skip_cv_pred,
            )
        else:
            print("\n[Skipping Lasso text feature analysis: run_lasso=False]")

        # =========================================================================
        # Stage 11: Additional Visualizations (Rankings + Wordclouds)
        # =========================================================================
        print("\n" + "="*70)
        print("STAGE 11: Additional Visualizations")
        print("="*70)

        from src.analysis.company_rankings import run_company_ranking_analysis
        from src.analysis.industry_rankings import run_industry_analysis
        from src.analysis.ai_wordclouds import run_ai_wordclouds

        run_company_ranking_analysis(
            f"{features_dir}/document_metrics.parquet",
            figures_dir,
            start_year=2020,
            end_year=2025
        )

        run_industry_analysis(
            doc_metrics_path=f"{features_dir}/document_metrics.parquet",
            final_dataset_path=input_dataset,
            output_dir=figures_dir,
            start_year=2020,
            end_year=2025,
            top_n=100
        )

        wordcloud_sample = dev_sample * 50 if dev_mode else None
        wordcloud_input = f"{features_dir}/sentences_with_keywords.parquet"
        run_ai_wordclouds(
            wordcloud_input,
            figures_dir,
            start_year=2020,
            end_year=2025,
            sample_n=wordcloud_sample
        )

        # =========================================================================
        # Summary
        # =========================================================================
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"Started: {start_time}")
        print(f"Finished: {end_time}")
        print(f"Duration: {duration}")
        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  - Features: {features_dir}/")
        print(f"  - Figures: {figures_dir}/")

        # Save run manifest
        output_csv = f"{output_dir}/pipeline_manifest.json"

        input_hash = sha256_file(input_dataset) if os.path.exists(input_dataset) else None
        wrds_hash = sha256_file(wrds_path) if os.path.exists(wrds_path) else None

        manifest = {
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration_seconds": duration.total_seconds(),
            "dev_mode": dev_mode,
            "dev_sample": dev_sample if dev_mode else None,
            "seed": seed,
            "ai_method": ai_method,
            "kw_workers": kw_workers,
            "metrics_workers": metrics_workers,
            "run_lasso": run_lasso,
            "lasso_max_features": lasso_max_features if run_lasso else None,
            "lasso_ngram_max": lasso_ngram_max if run_lasso else None,
            "lasso_cv": lasso_cv if run_lasso else None,
            "lasso_skip_cv_pred": lasso_skip_cv_pred if run_lasso else None,
            "git_head": get_git_head(),
            "log_path": log_path,
            "inputs": {
                "earnings_dataset": {"path": input_dataset, "sha256": input_hash},
                "wrds": {"path": wrds_path, "sha256": wrds_hash},
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "outputs": {
                "features_dir": features_dir,
                "figures_dir": figures_dir,
            },
        }

        with open(output_csv, 'w', encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S&P 500 AI Narrative Text Mining Pipeline"
    )
    
    parser.add_argument("--input", default="final_dataset.parquet",
                       help="Input earnings call dataset")
    parser.add_argument("--wrds", default="Sp500_meta_data.csv",
                       help="WRDS financial metadata")
    parser.add_argument("--output-dir", default="outputs",
                       help="Output directory")
    parser.add_argument("--dev", action="store_true",
                       help="Development mode (small samples)")
    parser.add_argument("--dev-sample", type=int, default=100,
                       help="Sample size for dev mode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--ai-method", default="topic", choices=["kw", "topic"],
                       help="AI detection method: kw (dictionary), topic (topic modeling)")
    parser.add_argument("--kw-workers", type=int, default=None,
                       help="Keyword detection workers (None = auto)")
    parser.add_argument("--metrics-workers", type=int, default=None,
                       help="AI intensity metrics workers (None = auto)")
    parser.add_argument("--skip-lasso", action="store_true",
                       help="Skip Lasso text-feature analysis stage (volcano/coefficient plots)")
    parser.add_argument("--lasso-max-features", type=int, default=5000,
                       help="Max TF-IDF features for Lasso text analysis")
    parser.add_argument("--lasso-ngram-max", type=int, default=2,
                       help="Max n-gram size for Lasso text analysis (min fixed at 1)")
    parser.add_argument("--lasso-cv", type=int, default=5,
                       help="Cross-validation folds for LassoCV")
    parser.add_argument("--lasso-skip-cv-pred", action="store_true",
                       help="Skip outer CV predictions/Kendall Tau scatter in Lasso stage for faster runs")
    
    args = parser.parse_args()
    
    run_pipeline(
        input_dataset=args.input,
        wrds_path=args.wrds,
        output_dir=args.output_dir,
        dev_mode=args.dev,
        dev_sample=args.dev_sample,
        seed=args.seed,
        ai_method=args.ai_method,
        kw_workers=args.kw_workers,
        metrics_workers=args.metrics_workers,
        run_lasso=not args.skip_lasso,
        lasso_max_features=args.lasso_max_features,
        lasso_ngram_max=args.lasso_ngram_max,
        lasso_cv=args.lasso_cv,
        lasso_skip_cv_pred=args.lasso_skip_cv_pred,
    )
