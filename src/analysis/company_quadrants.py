"""
Company Quadrants Module

Classifies companies into 4 types based on AI narrative patterns:
1. Aligned: High Speech + High Q&A (genuine AI focus)
2. Passive: Low Speech + High Q&A (responding to analyst pressure)
3. Self-Promoting: High Speech + Low Q&A (AI-washing)
4. Silent: Low Speech + Low Q&A (not engaging with AI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple
import os


def classify_companies(
    doc_metrics_df: pd.DataFrame,
    speech_col: str = 'speech_kw_ai_ratio',
    qa_col: str = 'qa_kw_ai_ratio',
    threshold_method: str = 'median'
) -> pd.DataFrame:
    """
    Classify companies into quadrants based on AI intensity.
    
    Args:
        doc_metrics_df: Document-level metrics
        speech_col: Column for speech AI intensity
        qa_col: Column for Q&A AI intensity
        threshold_method: 'median' or 'mean' for cutoff
        
    Returns:
        DataFrame with quadrant labels
    """
    df = doc_metrics_df.copy()
    
    # Compute thresholds
    if threshold_method == 'median':
        speech_threshold = df[speech_col].median()
        qa_threshold = df[qa_col].median()
    else:
        speech_threshold = df[speech_col].mean()
        qa_threshold = df[qa_col].mean()
    
    print(f"Thresholds: Speech={speech_threshold:.4f}, Q&A={qa_threshold:.4f}")
    
    # Classify
    def classify_row(row):
        high_speech = row[speech_col] >= speech_threshold
        high_qa = row[qa_col] >= qa_threshold
        
        if high_speech and high_qa:
            return 'Aligned'
        elif not high_speech and high_qa:
            return 'Passive'
        elif high_speech and not high_qa:
            return 'Self-Promoting'
        else:
            return 'Silent'
    
    df['quadrant'] = df.apply(classify_row, axis=1)
    
    return df, speech_threshold, qa_threshold


def aggregate_to_company(
    doc_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate document-level metrics to company level.
    
    Args:
        doc_metrics_df: Document-level data
        
    Returns:
        Company-level aggregated data
    """
    # Parse ticker from doc_id
    doc_metrics_df = doc_metrics_df.copy()
    doc_metrics_df['ticker'] = doc_metrics_df['doc_id'].apply(
        lambda x: str(x).rsplit('_', 1)[0] if '_' in str(x) else x
    )
    
    agg_df = doc_metrics_df.groupby('ticker').agg({
        'speech_kw_ai_ratio': 'mean',
        'qa_kw_ai_ratio': 'mean',
        'overall_kw_ai_ratio': 'mean',
        'doc_id': 'count'  # Number of earnings calls
    }).reset_index()
    
    agg_df = agg_df.rename(columns={'doc_id': 'num_calls'})
    
    return agg_df


def plot_quadrant_scatter(
    df: pd.DataFrame,
    speech_col: str,
    qa_col: str,
    speech_threshold: float,
    qa_threshold: float,
    output_path: str,
    title: str = "Company AI Narrative Quadrants"
):
    """
    Create scatter plot with quadrant visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by quadrant
    colors = {
        'Aligned': 'green',
        'Passive': 'blue',
        'Self-Promoting': 'orange',
        'Silent': 'gray'
    }
    
    for quadrant, color in colors.items():
        subset = df[df['quadrant'] == quadrant]
        ax.scatter(
            subset[speech_col], subset[qa_col],
            c=color, label=f"{quadrant} (n={len(subset)})",
            alpha=0.6, s=50
        )
    
    # Draw threshold lines
    ax.axvline(x=speech_threshold, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=qa_threshold, color='black', linestyle='--', alpha=0.5)
    
    # Labels for quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    offset = 0.02
    ax.text(xlim[1]-offset, ylim[1]-offset, 'Aligned\n(Genuine Focus)', 
            ha='right', va='top', fontsize=10, color='green', weight='bold')
    ax.text(xlim[0]+offset, ylim[1]-offset, 'Passive\n(Responding)', 
            ha='left', va='top', fontsize=10, color='blue', weight='bold')
    ax.text(xlim[1]-offset, ylim[0]+offset, 'Self-Promoting\n(AI-Washing?)', 
            ha='right', va='bottom', fontsize=10, color='orange', weight='bold')
    ax.text(xlim[0]+offset, ylim[0]+offset, 'Silent\n(Disengaged)', 
            ha='left', va='bottom', fontsize=10, color='gray', weight='bold')
    
    ax.set_xlabel('Speech AI Intensity (Management Prepared Remarks)', fontsize=12)
    ax.set_ylabel('Q&A AI Intensity (Analyst Interaction)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quadrant plot to {output_path}")


def plot_quadrant_distribution(
    df: pd.DataFrame,
    output_path: str
):
    """
    Create bar chart of quadrant distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df['quadrant'].value_counts()
    colors = ['green', 'blue', 'orange', 'gray']
    order = ['Aligned', 'Passive', 'Self-Promoting', 'Silent']
    
    counts = counts.reindex(order)
    
    bars = ax.bar(counts.index, counts.values, color=colors)
    
    # Add count labels
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel('Quadrant', fontsize=12)
    ax.set_ylabel('Number of Companies', fontsize=12)
    ax.set_title('Distribution of Companies by AI Narrative Pattern', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {output_path}")


def run_quadrant_analysis(
    doc_metrics_path: str,
    output_dir: str = "outputs/figures"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full quadrant analysis pipeline.
    
    Args:
        doc_metrics_path: Path to document metrics
        output_dir: Output directory
        
    Returns:
        Tuple of (document-level df, company-level df)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading metrics...")
    doc_metrics = pd.read_parquet(doc_metrics_path)
    
    # Document-level analysis
    print("\nDocument-level quadrant analysis...")
    doc_classified, speech_th, qa_th = classify_companies(doc_metrics)
    
    print("\nQuadrant Distribution (Documents):")
    print(doc_classified['quadrant'].value_counts())
    
    plot_quadrant_scatter(
        doc_classified, 'speech_kw_ai_ratio', 'qa_kw_ai_ratio',
        speech_th, qa_th,
        f"{output_dir}/quadrant_scatter_documents.png",
        "Document-Level AI Narrative Quadrants"
    )
    
    # Company-level analysis
    print("\nCompany-level quadrant analysis...")
    company_agg = aggregate_to_company(doc_metrics)
    company_classified, comp_speech_th, comp_qa_th = classify_companies(company_agg)
    
    print("\nQuadrant Distribution (Companies):")
    print(company_classified['quadrant'].value_counts())
    
    plot_quadrant_scatter(
        company_classified, 'speech_kw_ai_ratio', 'qa_kw_ai_ratio',
        comp_speech_th, comp_qa_th,
        f"{output_dir}/quadrant_scatter_companies.png",
        "Company-Level AI Narrative Quadrants"
    )
    
    plot_quadrant_distribution(company_classified, f"{output_dir}/quadrant_distribution.png")
    
    # Save results
    doc_classified.to_parquet(f"{output_dir}/../features/documents_with_quadrants.parquet", index=False)
    company_classified.to_csv(f"{output_dir}/company_quadrants.csv", index=False)
    
    return doc_classified, company_classified


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quadrant analysis")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--output-dir", default="outputs/figures")
    
    args = parser.parse_args()
    
    run_quadrant_analysis(args.metrics, args.output_dir)
