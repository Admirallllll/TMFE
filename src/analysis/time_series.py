"""
Time Series Analysis Module

Analyzes AI narrative trends over time (Speech vs Q&A intensity).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
from datetime import datetime
import os


def prepare_time_series_data(
    doc_metrics_df: pd.DataFrame,
    final_dataset_df: pd.DataFrame,
    doc_id_col: str = 'doc_id'
) -> pd.DataFrame:
    """
    Merge document metrics with date information.
    
    Args:
        doc_metrics_df: Document-level AI metrics
        final_dataset_df: Original dataset with dates
        
    Returns:
        Merged DataFrame with time info
    """
    # Extract ticker/year/quarter from doc_id (format: TICKER_YYYYQX)
    doc_metrics_df = doc_metrics_df.copy()
    
    def parse_doc_id(doc_id):
        parts = str(doc_id).rsplit('_', 1)
        if len(parts) == 2:
            ticker = parts[0]
            yq = parts[1]
            if 'Q' in yq:
                year = int(yq.split('Q')[0])
                quarter = int(yq.split('Q')[1])
                return ticker, year, quarter
        return None, None, None
    
    parsed = doc_metrics_df[doc_id_col].apply(parse_doc_id)
    doc_metrics_df['ticker'] = [p[0] for p in parsed]
    doc_metrics_df['year'] = [p[1] for p in parsed]
    doc_metrics_df['quarter'] = [p[2] for p in parsed]

    # Prefer the true earnings call date from the input dataset (more accurate than a quarter midpoint)
    date_map = (
        final_dataset_df[['ticker', 'year', 'quarter', 'date']]
        .copy()
    )
    date_map['date'] = pd.to_datetime(date_map['date'], errors='coerce')
    date_map = (
        date_map.dropna(subset=['ticker', 'year', 'quarter', 'date'])
        .groupby(['ticker', 'year', 'quarter'], as_index=False)['date']
        .min()
    )

    doc_metrics_df = doc_metrics_df.merge(
        date_map,
        on=['ticker', 'year', 'quarter'],
        how='left'
    )

    # Fallback to an approximate date if mapping fails (should be rare)
    missing_date = doc_metrics_df['date'].isna()
    if missing_date.any():
        def quarter_to_date(row):
            if pd.isna(row['year']) or pd.isna(row['quarter']):
                return None
            month = (row['quarter'] - 1) * 3 + 2  # Middle of quarter
            return datetime(int(row['year']), month, 15)

        doc_metrics_df.loc[missing_date, 'date'] = doc_metrics_df.loc[missing_date].apply(quarter_to_date, axis=1)

    doc_metrics_df['year_quarter'] = doc_metrics_df['year'].astype(str) + 'Q' + doc_metrics_df['quarter'].astype(str)
    
    return doc_metrics_df


def compute_aggregate_trends(
    df: pd.DataFrame,
    group_col: str = 'year_quarter'
) -> pd.DataFrame:
    """
    Compute aggregate AI intensity trends over time.
    
    Args:
        df: DataFrame with document-level metrics
        group_col: Column to group by (default: year_quarter)
        
    Returns:
        DataFrame with time-aggregated metrics
    """
    agg_df = df.groupby(group_col).agg({
        'speech_ml_ai_ratio': ['mean', 'std', 'count'],
        'qa_ml_ai_ratio': ['mean', 'std'],
        'speech_kw_ai_ratio': ['mean', 'std'],
        'qa_kw_ai_ratio': ['mean', 'std'],
        'overall_ml_ai_ratio': 'mean'
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns]
    
    return agg_df


def plot_ai_trends(
    trend_df: pd.DataFrame,
    output_path: str,
    title: str = "AI Narrative Intensity Over Time"
):
    """
    Plot Speech vs Q&A AI intensity trends.
    
    Args:
        trend_df: Aggregated trend data
        output_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Sort by time
    trend_df = trend_df.sort_values('year_quarter')
    x = range(len(trend_df))
    
    # Plot 1: ML-based AI intensity
    ax1 = axes[0]
    ax1.plot(x, trend_df['speech_ml_ai_ratio_mean'], 'b-o', label='Speech AI Intensity', linewidth=2)
    ax1.plot(x, trend_df['qa_ml_ai_ratio_mean'], 'r-s', label='Q&A AI Intensity', linewidth=2)
    
    # Add ChatGPT release marker (Nov 2022 = 2022Q4)
    chatgpt_idx = None
    for i, yq in enumerate(trend_df['year_quarter']):
        if '2022Q4' in str(yq):
            chatgpt_idx = i
            break
    
    if chatgpt_idx is not None:
        ax1.axvline(x=chatgpt_idx, color='green', linestyle='--', alpha=0.7, label='ChatGPT Release (Nov 2022)')
    
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('AI Intensity (% of sentences)')
    ax1.set_title(f'{title} (Transfer Learning Model)')
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(trend_df['year_quarter'].iloc[::2], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Keyword-based (baseline) comparison
    ax2 = axes[1]
    ax2.plot(x, trend_df['speech_kw_ai_ratio_mean'], 'b--o', label='Speech AI (Keywords)', alpha=0.7)
    ax2.plot(x, trend_df['qa_kw_ai_ratio_mean'], 'r--s', label='Q&A AI (Keywords)', alpha=0.7)
    
    if chatgpt_idx is not None:
        ax2.axvline(x=chatgpt_idx, color='green', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('AI Intensity (% of sentences)')
    ax2.set_title('AI Intensity Over Time (Keyword Baseline)')
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels(trend_df['year_quarter'].iloc[::2], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trend plot to {output_path}")


def plot_method_comparison(
    trend_df: pd.DataFrame,
    output_path: str
):
    """
    Compare ML vs Keyword detection over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    trend_df = trend_df.sort_values('year_quarter')
    x = range(len(trend_df))
    
    # Plot difference (ML - Keyword)
    speech_diff = trend_df['speech_ml_ai_ratio_mean'] - trend_df['speech_kw_ai_ratio_mean']
    qa_diff = trend_df['qa_ml_ai_ratio_mean'] - trend_df['qa_kw_ai_ratio_mean']
    
    ax.bar([i - 0.2 for i in x], speech_diff, width=0.35, label='Speech (ML - KW)', color='blue', alpha=0.7)
    ax.bar([i + 0.2 for i in x], qa_diff, width=0.35, label='Q&A (ML - KW)', color='red', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Detection Difference (ML - Keyword)')
    ax.set_title('Transfer Learning vs Keyword Detection: Additional AI Content Detected')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(trend_df['year_quarter'].iloc[::2], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def run_time_series_analysis(
    doc_metrics_path: str,
    final_dataset_path: str,
    output_dir: str = "outputs/figures"
) -> pd.DataFrame:
    """
    Full time series analysis pipeline.
    
    Args:
        doc_metrics_path: Path to document metrics
        final_dataset_path: Path to original dataset
        output_dir: Directory for figures
        
    Returns:
        Trend DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    doc_metrics = pd.read_parquet(doc_metrics_path)
    final_dataset = pd.read_parquet(final_dataset_path)
    
    print("Preparing time series data...")
    ts_df = prepare_time_series_data(doc_metrics, final_dataset)
    
    print("Computing aggregate trends...")
    trends = compute_aggregate_trends(ts_df)
    
    print("Generating plots...")
    plot_ai_trends(trends, f"{output_dir}/ai_trends_over_time.png")
    plot_method_comparison(trends, f"{output_dir}/ml_vs_keyword_comparison.png")
    
    # Save trends
    trends.to_csv(f"{output_dir}/ai_trends.csv", index=False)
    
    return trends


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Time series analysis")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--dataset", default="final_dataset.parquet")
    parser.add_argument("--output-dir", default="outputs/figures")
    
    args = parser.parse_args()
    
    run_time_series_analysis(args.metrics, args.dataset, args.output_dir)
