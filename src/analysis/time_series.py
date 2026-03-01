"""
Time Series Analysis Module

Analyzes AI narrative trends over time (Speech vs Q&A intensity).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from datetime import datetime
import os

try:
    from src.utils.visual_style import SPOTIFY_COLORS, apply_spotify_theme, save_figure, style_axes, style_legend
except Exception:  # pragma: no cover
    SPOTIFY_COLORS = {"background": "#121212", "blue": "#4EA1FF", "negative": "#FF5A5F", "accent": "#1DB954"}
    def apply_spotify_theme():
        return None
    def style_axes(ax, **kwargs):
        return ax
    def style_legend(ax):
        return ax.get_legend()
    def save_figure(fig, output_path: str, dpi: int = 150):
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


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
    # Create indicator for whether a document mentions AI at all
    df = df.copy()
    df['mentions_ai'] = (df['overall_kw_ai_ratio'] > 0).astype(int)

    agg_df = df.groupby(group_col).agg({
        'speech_kw_ai_ratio': ['mean', 'std', 'count'],
        'qa_kw_ai_ratio': ['mean', 'std'],
        'overall_kw_ai_ratio': 'mean',
        'mentions_ai': 'mean'
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns]
    
    # Rename mentions_ai_mean to something more descriptive
    agg_df = agg_df.rename(columns={'mentions_ai_mean': 'prop_docs_mentioning_ai'})
    
    return agg_df


def plot_ai_trends(
    trend_df: pd.DataFrame,
    output_path: str,
    title: str = "AI Narrative Intensity Over Time"
):
    """
    Plot Speech vs Q&A AI intensity trends matching the reference style.
    
    Args:
        trend_df: Aggregated trend data
        output_path: Path to save figure
        title: Plot title
    """
    # Use standard white background style
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Sort by time
    trend_df = trend_df.sort_values('year_quarter')
    x = range(len(trend_df))
    
    # Plot 1: Speech AI Intensity (Blue with circles)
    line1, = ax1.plot(x, trend_df['speech_kw_ai_ratio_mean'], 
                     color='blue', marker='o', linestyle='-', 
                     linewidth=2, label='Speech AI Intensity')
    
    # Plot 2: Q&A AI Intensity (Red with squares)
    line2, = ax1.plot(x, trend_df['qa_kw_ai_ratio_mean'], 
                     color='red', marker='s', linestyle='-', 
                     linewidth=2, label='Q&A AI Intensity')
    
    # Plot 3: Proportion of documents mentioning AI (Secondary Y-axis, dashed green)
    # Keeping this as it was requested as a supplemental metric
    ax2 = ax1.twinx()
    line3, = ax2.plot(x, trend_df['prop_docs_mentioning_ai'], 
                     color='green', marker='d', linestyle='--', 
                     linewidth=1, alpha=0.5, label='Proportion of AI Calls')
    
    # Add ChatGPT release vertical line (Green dashed)
    chatgpt_idx = None
    for i, yq in enumerate(trend_df['year_quarter']):
        if '2022Q4' in str(yq):
            chatgpt_idx = i
            break
    
    chatgpt_line = None
    if chatgpt_idx is not None:
        chatgpt_line = ax1.axvline(x=chatgpt_idx, color='green', linestyle='--', 
                                  linewidth=1, label='ChatGPT Release (Nov 2022)')
    
    # Grid, Labels and Title
    ax1.grid(True, linestyle='-', alpha=0.2)
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('AI Intensity (% of sentences)')
    ax2.set_ylabel('Share of Documents Mentioning AI')
    ax1.set_title(f'{title} (Dictionary)')
    
    # X-axis ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(trend_df['year_quarter'], rotation=45, ha='right')
    
    # Legend - matching the reference image style (top left)
    # We include Speech, Q&A, and ChatGPT Release in the main legend
    legend_elements = [line1, line2]
    if chatgpt_line:
        legend_elements.append(chatgpt_line)
    legend_elements.append(line3) # Also include the new metric
    
    ax1.legend(handles=legend_elements, loc='upper left', frameon=True)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trend plot to {output_path}")





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
    
    # Save trends
    trends.to_csv(f"{output_dir}/ai_trends.csv", index=False)
    
    return trends


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze AI topic intensity over time")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--dataset", default="data/final_dataset.parquet")
    parser.add_argument("--output", default="outputs/figures")
    
    args = parser.parse_args()
    
    run_time_series_analysis(args.metrics, args.dataset, args.output)
