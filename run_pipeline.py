"""
Main Pipeline Script

End-to-end orchestration of the S&P 500 AI Narrative Text Mining project.

Stages:
1. Parse transcripts (Speech/Q&A splitting)
2. Split into sentences
3. Prepare AI news training data
4. Train FinBERT classifier
5. Run inference on transcript sentences
6. Compute keyword baseline
7. Compute AI intensity metrics
8. Compute initiation scores
9. Run analyses (time series, quadrants, regression)
"""

import os
import argparse
from datetime import datetime
import json
import hashlib
import platform
import subprocess
import sys


def run_pipeline(
    input_dataset: str = "final_dataset.parquet",
    ai_news_path: str = "ai_media_dataset_20250911.csv",
    wrds_path: str = "Sp500_meta_data.csv",
    output_dir: str = "outputs",
    model_epochs: int = 3,
    batch_size: int = 16,
    dev_mode: bool = False,
    dev_sample: int = 100,
    skip_training: bool = False,
    seed: int = 42
):
    """
    Run the full pipeline.
    
    Args:
        input_dataset: Path to earnings call dataset
        ai_news_path: Path to AI news dataset
        wrds_path: Path to WRDS metadata
        output_dir: Output directory
        model_epochs: Number of training epochs
        batch_size: Batch size
        dev_mode: Development mode (small samples)
        dev_sample: Sample size for dev mode
        skip_training: Skip model training (use existing model)
        seed: Random seed for reproducibility
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
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    start_time = datetime.now()
    print("="*70)
    print("S&P 500 AI Narrative Text Mining Pipeline")
    print(f"Started at: {start_time}")
    print("="*70)
    
    # Create output directories
    features_dir = f"{output_dir}/features"
    models_dir = f"{output_dir}/models"
    figures_dir = f"{output_dir}/figures"
    
    for d in [features_dir, models_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)
    
    sample_n = dev_sample if dev_mode else None

    # Validate input dataset schema early (high-stakes fail-fast)
    import pandas as pd
    required_cols = {"ticker", "date", "year", "quarter", "structured_content"}
    try:
        import pyarrow.parquet as pq

        ds_cols = set(pq.ParquetFile(input_dataset).schema.names)
    except Exception:
        # Fallback (less efficient): read minimal columns via pandas
        ds_cols = set(pd.read_parquet(input_dataset, columns=["ticker"]).columns.tolist())
    missing = sorted(required_cols - ds_cols)
    if missing:
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
    sentences_with_kw = compute_keyword_metrics(sentences_df)
    sentences_with_kw.to_parquet(f"{features_dir}/sentences_with_keywords.parquet", index=False)
    
    # =========================================================================
    # Stage 4: Prepare AI News Training Data
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 4: Prepare AI News Training Data")
    print("="*70)
    
    from src.models.ai_news_dataset import prepare_training_data
    
    train_path = f"{features_dir}/ai_news_train.parquet"
    if not os.path.exists(train_path) or dev_mode:
        ai_news_sample = dev_sample * 50 if dev_mode else None  # Need more data for training
        prepare_training_data(ai_news_path, features_dir, ai_news_sample)
    else:
        print(f"Using existing training data: {train_path}")
    
    # =========================================================================
    # Stage 5: Train FinBERT Classifier
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 5: Train FinBERT Classifier")
    print("="*70)
    
    model_path = f"{models_dir}/best_model"
    
    if not skip_training and (not os.path.exists(model_path) or dev_mode):
        from src.models.ai_classifier import train_classifier
        
        train_sample = dev_sample * 10 if dev_mode else None
        epochs = 1 if dev_mode else model_epochs
        
        train_classifier(
            f"{features_dir}/ai_news_train.parquet",
            f"{features_dir}/ai_news_val.parquet",
            models_dir,
            epochs,
            batch_size,
            train_sample
        )
    else:
        print(f"Using existing model: {model_path}")
    
    # =========================================================================
    # Stage 6: Run Inference on Transcript Sentences
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 6: Run Inference on Transcript Sentences")
    print("="*70)
    
    from src.models.predict import run_inference_pipeline
    
    predictions_path = f"{features_dir}/sentences_with_predictions.parquet"
    
    # Merge keyword results with ML predictions
    run_inference_pipeline(
        f"{features_dir}/sentences_with_keywords.parquet",
        model_path,
        predictions_path,
        batch_size
    )
    
    # =========================================================================
    # Stage 7: Compute AI Intensity Metrics
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 7: Compute AI Intensity Metrics")
    print("="*70)
    
    from src.metrics.ai_intensity import compute_all_metrics
    
    sentences_with_pred = pd.read_parquet(predictions_path)
    compute_all_metrics(sentences_with_pred, features_dir)
    
    # =========================================================================
    # Stage 8: Compute Initiation Scores
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 8: Compute AI Initiation Scores")
    print("="*70)
    
    from src.metrics.initiation_score import compute_all_initiation_metrics
    
    compute_all_initiation_metrics(sentences_with_pred, features_dir)
    
    # =========================================================================
    # Stage 9: Analysis - Time Series
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 9: Time Series Analysis")
    print("="*70)
    
    from src.analysis.time_series import run_time_series_analysis
    
    run_time_series_analysis(
        f"{features_dir}/document_metrics.parquet",
        input_dataset,
        figures_dir
    )
    
    # =========================================================================
    # Stage 10: Analysis - Company Quadrants
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 10: Company Quadrant Analysis")
    print("="*70)
    
    from src.analysis.company_quadrants import run_quadrant_analysis
    
    run_quadrant_analysis(
        f"{features_dir}/document_metrics.parquet",
        figures_dir
    )
    
    # =========================================================================
    # Stage 11: Regression Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE 11: Regression Analysis")
    print("="*70)
    
    from src.analysis.regression import run_regression_analysis
    
    run_regression_analysis(
        f"{features_dir}/initiation_scores.parquet",
        f"{features_dir}/document_metrics.parquet",
        wrds_path,
        figures_dir
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
    print(f"  - Models: {models_dir}/")
    print(f"  - Figures: {figures_dir}/")
    
    # Save run manifest
    output_csv = f"{output_dir}/pipeline_manifest.json"

    input_hash = sha256_file(input_dataset) if os.path.exists(input_dataset) else None
    ai_news_hash = sha256_file(ai_news_path) if os.path.exists(ai_news_path) else None
    wrds_hash = sha256_file(wrds_path) if os.path.exists(wrds_path) else None

    manifest = {
        "start_time": str(start_time),
        "end_time": str(end_time),
        "duration_seconds": duration.total_seconds(),
        "dev_mode": dev_mode,
        "dev_sample": dev_sample if dev_mode else None,
        "seed": seed,
        "model_epochs": model_epochs,
        "batch_size": batch_size,
        "git_head": get_git_head(),
        "inputs": {
            "earnings_dataset": {"path": input_dataset, "sha256": input_hash},
            "ai_news": {"path": ai_news_path, "sha256": ai_news_hash},
            "wrds": {"path": wrds_path, "sha256": wrds_hash},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "outputs": {
            "features_dir": features_dir,
            "models_dir": models_dir,
            "figures_dir": figures_dir,
        },
    }
    
    with open(output_csv, 'w', encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S&P 500 AI Narrative Text Mining Pipeline"
    )
    
    parser.add_argument("--input", default="final_dataset.parquet",
                       help="Input earnings call dataset")
    parser.add_argument("--ai-news", default="ai_media_dataset_20250911.csv",
                       help="AI news dataset for training")
    parser.add_argument("--wrds", default="Sp500_meta_data.csv",
                       help="WRDS financial metadata")
    parser.add_argument("--output-dir", default="outputs",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--dev", action="store_true",
                       help="Development mode (small samples)")
    parser.add_argument("--dev-sample", type=int, default=100,
                       help="Sample size for dev mode")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    run_pipeline(
        input_dataset=args.input,
        ai_news_path=args.ai_news,
        wrds_path=args.wrds,
        output_dir=args.output_dir,
        model_epochs=args.epochs,
        batch_size=args.batch_size,
        dev_mode=args.dev,
        dev_sample=args.dev_sample,
        skip_training=args.skip_training,
        seed=args.seed
    )
