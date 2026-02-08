"""
AI Intensity Metrics Module

Computes AI intensity scores at the document and section level.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_section_intensity(
    sentences_df: pd.DataFrame,
    doc_id_col: str = 'doc_id',
    section_col: str = 'section',
    ml_pred_col: str = 'ml_is_ai',
    ml_prob_col: str = 'ml_ai_prob',
    kw_pred_col: str = 'kw_is_ai'
) -> pd.DataFrame:
    """
    Compute AI intensity metrics per document and section.
    
    Args:
        sentences_df: DataFrame with sentence-level predictions
        doc_id_col: Column for document ID
        section_col: Column for section (speech/qa)
        ml_pred_col: Column for ML model predictions
        ml_prob_col: Column for ML probabilities
        kw_pred_col: Column for keyword predictions
        
    Returns:
        DataFrame with document-section level metrics
    """
    results = []
    
    for doc_id in tqdm(sentences_df[doc_id_col].unique(), desc="Computing intensities"):
        doc_df = sentences_df[sentences_df[doc_id_col] == doc_id]
        
        for section in ['speech', 'qa']:
            section_df = doc_df[doc_df[section_col] == section]
            
            if len(section_df) == 0:
                continue
            
            result = {
                'doc_id': doc_id,
                'section': section,
                'total_sentences': len(section_df),
            }
            
            # ML-based metrics
            if ml_pred_col in section_df.columns:
                result['ml_ai_sentences'] = section_df[ml_pred_col].sum()
                result['ml_ai_ratio'] = section_df[ml_pred_col].mean()
                result['ml_avg_prob'] = section_df[ml_prob_col].mean()
                result['ml_max_prob'] = section_df[ml_prob_col].max()
            
            # Keyword-based metrics
            if kw_pred_col in section_df.columns:
                result['kw_ai_sentences'] = section_df[kw_pred_col].sum()
                result['kw_ai_ratio'] = section_df[kw_pred_col].mean()
            
            results.append(result)
    
    return pd.DataFrame(results)


def compute_document_intensity(
    section_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate section metrics to document level.
    
    Args:
        section_metrics_df: DataFrame from compute_section_intensity
        
    Returns:
        DataFrame with document-level metrics
    """
    results = []
    
    for doc_id in section_metrics_df['doc_id'].unique():
        doc_df = section_metrics_df[section_metrics_df['doc_id'] == doc_id]
        
        result = {'doc_id': doc_id}
        
        for section in ['speech', 'qa']:
            section_row = doc_df[doc_df['section'] == section]
            
            if len(section_row) == 0:
                result[f'{section}_total_sentences'] = 0
                result[f'{section}_ml_ai_ratio'] = 0.0
                result[f'{section}_kw_ai_ratio'] = 0.0
            else:
                row = section_row.iloc[0]
                result[f'{section}_total_sentences'] = row.get('total_sentences', 0)
                result[f'{section}_ml_ai_ratio'] = row.get('ml_ai_ratio', 0.0)
                result[f'{section}_ml_ai_sentences'] = row.get('ml_ai_sentences', 0)
                result[f'{section}_kw_ai_ratio'] = row.get('kw_ai_ratio', 0.0)
                result[f'{section}_kw_ai_sentences'] = row.get('kw_ai_sentences', 0)
                result[f'{section}_ml_avg_prob'] = row.get('ml_avg_prob', 0.0)
        
        # Compute overall metrics
        total_sents = result.get('speech_total_sentences', 0) + result.get('qa_total_sentences', 0)
        if total_sents > 0:
            ml_ai_total = result.get('speech_ml_ai_sentences', 0) + result.get('qa_ml_ai_sentences', 0)
            result['overall_ml_ai_ratio'] = ml_ai_total / total_sents
            
            kw_ai_total = result.get('speech_kw_ai_sentences', 0) + result.get('qa_kw_ai_sentences', 0)
            result['overall_kw_ai_ratio'] = kw_ai_total / total_sents
        else:
            result['overall_ml_ai_ratio'] = 0.0
            result['overall_kw_ai_ratio'] = 0.0
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_all_metrics(
    sentences_df: pd.DataFrame,
    output_dir: str = "outputs/features"
) -> Dict[str, pd.DataFrame]:
    """
    Compute all AI intensity metrics and save.
    
    Args:
        sentences_df: Sentence-level data with predictions
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with metric DataFrames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Computing section-level metrics...")
    section_metrics = compute_section_intensity(sentences_df)
    section_metrics.to_parquet(f"{output_dir}/section_metrics.parquet", index=False)
    
    print("Computing document-level metrics...")
    doc_metrics = compute_document_intensity(section_metrics)
    doc_metrics.to_parquet(f"{output_dir}/document_metrics.parquet", index=False)
    
    print(f"\n=== AI Intensity Summary ===")
    print(f"Documents analyzed: {len(doc_metrics)}")
    print(f"Avg Speech AI Ratio (ML): {doc_metrics['speech_ml_ai_ratio'].mean():.3f}")
    print(f"Avg Q&A AI Ratio (ML): {doc_metrics['qa_ml_ai_ratio'].mean():.3f}")
    print(f"Avg Speech AI Ratio (KW): {doc_metrics['speech_kw_ai_ratio'].mean():.3f}")
    print(f"Avg Q&A AI Ratio (KW): {doc_metrics['qa_kw_ai_ratio'].mean():.3f}")
    
    return {
        'section_metrics': section_metrics,
        'document_metrics': doc_metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute AI intensity metrics")
    parser.add_argument("--input", default="outputs/features/sentences_with_predictions.parquet")
    parser.add_argument("--output-dir", default="outputs/features")
    
    args = parser.parse_args()
    
    sentences_df = pd.read_parquet(args.input)
    compute_all_metrics(sentences_df, args.output_dir)
