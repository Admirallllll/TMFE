"""
Prediction Module

Batch inference on earnings transcript sentences using trained AI classifier.
"""

import os
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch


def predict_sentences(
    sentences_df: pd.DataFrame,
    model_path: str,
    text_col: str = 'text',
    batch_size: int = 32,
    device: str = None
) -> pd.DataFrame:
    """
    Apply AI classifier to sentence dataset.
    
    Args:
        sentences_df: DataFrame with sentences
        model_path: Path to trained model
        text_col: Column containing text
        batch_size: Batch size for inference
        device: Device for inference (cuda/cpu)
        
    Returns:
        DataFrame with added prediction columns
    """
    from .ai_classifier import AIClassifier
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    classifier = AIClassifier.load(model_path, device=device)
    
    print(f"Running inference on {len(sentences_df)} sentences...")
    texts = sentences_df[text_col].tolist()
    
    predictions, probabilities = classifier.predict(texts, batch_size=batch_size)
    
    # Add predictions to dataframe
    result_df = sentences_df.copy()
    result_df['ml_is_ai'] = predictions
    result_df['ml_ai_prob'] = probabilities
    
    # Summary stats
    ai_count = predictions.sum()
    print(f"\n=== Prediction Summary ===")
    print(f"Total sentences: {len(predictions)}")
    print(f"AI-related: {ai_count} ({ai_count/len(predictions)*100:.1f}%)")
    print(f"Avg AI probability: {probabilities.mean():.3f}")
    
    return result_df


def run_inference_pipeline(
    sentences_path: str,
    model_path: str,
    output_path: str,
    batch_size: int = 32,
    device: str = None
) -> pd.DataFrame:
    """
    Full inference pipeline.
    
    Args:
        sentences_path: Path to sentences parquet
        model_path: Path to trained model
        output_path: Path to save predictions
        batch_size: Batch size
        device: Device for inference
        
    Returns:
        DataFrame with predictions
    """
    print(f"Loading sentences from {sentences_path}...")
    sentences_df = pd.read_parquet(sentences_path)
    
    result_df = predict_sentences(
        sentences_df,
        model_path,
        batch_size=batch_size,
        device=device
    )
    
    print(f"\nSaving predictions to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    
    return result_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run AI detection inference")
    parser.add_argument("--sentences", default="outputs/features/sentences.parquet")
    parser.add_argument("--model", default="outputs/models/best_model")
    parser.add_argument("--output", default="outputs/features/sentences_with_predictions.parquet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    
    args = parser.parse_args()
    
    run_inference_pipeline(
        args.sentences,
        args.model,
        args.output,
        args.batch_size,
        args.device
    )
