"""
AI Initiation Score Module

Computes who initiates AI discussions in Q&A sessions:
- Analyst-initiated: AI topic first raised by analyst question
- Management-initiated: AI topic introduced by management (in response to unrelated question)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class QAExchange:
    """Represents a Q&A exchange (question + answer pair)."""
    doc_id: str
    exchange_idx: int
    question_text: str
    answer_text: str
    questioner: str
    answerer: str
    question_is_ai: bool
    answer_is_ai: bool
    question_ai_prob: float
    answer_ai_prob: float


def extract_qa_exchanges(
    sentences_df: pd.DataFrame,
    doc_id_col: str = 'doc_id',
    section_col: str = 'section',
    role_col: str = 'role',
    turn_idx_col: str = 'turn_idx',
    text_col: str = 'text',
    ml_pred_col: str = 'ml_is_ai',
    ml_prob_col: str = 'ml_ai_prob'
) -> List[QAExchange]:
    """
    Extract Q&A exchanges from sentence data.
    
    An exchange is a question followed by answer(s).
    
    Args:
        sentences_df: Sentence-level data
        
    Returns:
        List of QAExchange objects
    """
    exchanges = []
    
    # Filter to Q&A section only
    qa_df = sentences_df[sentences_df[section_col] == 'qa'].copy()
    
    if len(qa_df) == 0:
        return exchanges
    
    for doc_id in qa_df[doc_id_col].unique():
        doc_df = qa_df[qa_df[doc_id_col] == doc_id].sort_values(turn_idx_col)
        
        # Group by turn
        turns = []
        for turn_idx in doc_df[turn_idx_col].unique():
            turn_df = doc_df[doc_df[turn_idx_col] == turn_idx]
            if len(turn_df) > 0:
                # Aggregate turn info
                turn_text = ' '.join(turn_df[text_col].tolist())
                role = turn_df[role_col].iloc[0]
                
                # Check if any sentence is AI-related
                is_ai = turn_df[ml_pred_col].any() if ml_pred_col in turn_df.columns else False
                avg_prob = turn_df[ml_prob_col].mean() if ml_prob_col in turn_df.columns else 0.0
                
                turns.append({
                    'turn_idx': turn_idx,
                    'text': turn_text,
                    'role': role,
                    'is_ai': is_ai,
                    'ai_prob': avg_prob,
                    'speaker': turn_df['speaker'].iloc[0] if 'speaker' in turn_df.columns else ''
                })
        
        # Match questions with answers
        exchange_idx = 0
        i = 0
        while i < len(turns):
            turn = turns[i]
            
            # Look for analyst turn (question)
            if turn['role'] == 'analyst':
                question = turn
                
                # Find following management answer(s)
                answers = []
                j = i + 1
                while j < len(turns) and turns[j]['role'] in ['management', 'unknown']:
                    answers.append(turns[j])
                    j += 1
                
                if answers:
                    # Combine answer texts
                    answer_text = ' '.join([a['text'] for a in answers])
                    answer_is_ai = any(a['is_ai'] for a in answers)
                    answer_ai_prob = max(a['ai_prob'] for a in answers)
                    
                    exchanges.append(QAExchange(
                        doc_id=doc_id,
                        exchange_idx=exchange_idx,
                        question_text=question['text'],
                        answer_text=answer_text,
                        questioner=question['speaker'],
                        answerer=answers[0]['speaker'],
                        question_is_ai=question['is_ai'],
                        answer_is_ai=answer_is_ai,
                        question_ai_prob=question['ai_prob'],
                        answer_ai_prob=answer_ai_prob
                    ))
                    exchange_idx += 1
                    i = j
                    continue
            
            i += 1
    
    return exchanges


def compute_initiation_scores(
    exchanges: List[QAExchange]
) -> pd.DataFrame:
    """
    Compute AI initiation scores per document.
    
    Metrics:
    - analyst_initiated_ratio: % of AI discussions started by analyst question
    - management_pivot_ratio: % of AI discussions where management introduced AI
      in response to non-AI question
    - total_ai_exchanges: Total exchanges involving AI
    
    Args:
        exchanges: List of QAExchange objects
        
    Returns:
        DataFrame with per-document initiation scores
    """
    if not exchanges:
        return pd.DataFrame()
    
    # Convert to DataFrame
    exchange_df = pd.DataFrame([{
        'doc_id': e.doc_id,
        'exchange_idx': e.exchange_idx,
        'question_is_ai': e.question_is_ai,
        'answer_is_ai': e.answer_is_ai,
        'question_ai_prob': e.question_ai_prob,
        'answer_ai_prob': e.answer_ai_prob
    } for e in exchanges])
    
    results = []
    
    for doc_id in exchange_df['doc_id'].unique():
        doc_df = exchange_df[exchange_df['doc_id'] == doc_id]
        
        total_exchanges = len(doc_df)
        
        # AI-related exchanges (either question or answer mentions AI)
        ai_exchanges = doc_df[(doc_df['question_is_ai']) | (doc_df['answer_is_ai'])]
        total_ai_exchanges = len(ai_exchanges)
        
        if total_ai_exchanges == 0:
            results.append({
                'doc_id': doc_id,
                'total_exchanges': total_exchanges,
                'total_ai_exchanges': 0,
                'analyst_initiated_count': 0,
                'management_pivot_count': 0,
                'mutual_ai_count': 0,
                'analyst_initiated_ratio': 0.0,
                'management_pivot_ratio': 0.0,
                'ai_initiation_score': 0.5  # Neutral
            })
            continue
        
        # Analyst initiated: Question is AI, answer is AI
        analyst_initiated = doc_df[(doc_df['question_is_ai']) & (doc_df['answer_is_ai'])]
        analyst_initiated_count = len(analyst_initiated)
        
        # Management pivot: Question is NOT AI, but answer IS AI
        management_pivot = doc_df[(~doc_df['question_is_ai']) & (doc_df['answer_is_ai'])]
        management_pivot_count = len(management_pivot)
        
        # Question is AI but answer is not (analyst trying, management deflecting)
        analyst_only = doc_df[(doc_df['question_is_ai']) & (~doc_df['answer_is_ai'])]
        
        # AI Initiation Score: Higher = more management-driven
        # Score = management_pivot / (analyst_initiated + management_pivot)
        denom = analyst_initiated_count + management_pivot_count
        if denom > 0:
            ai_initiation_score = management_pivot_count / denom
        else:
            ai_initiation_score = 0.5
        
        results.append({
            'doc_id': doc_id,
            'total_exchanges': total_exchanges,
            'total_ai_exchanges': total_ai_exchanges,
            'analyst_initiated_count': analyst_initiated_count,
            'management_pivot_count': management_pivot_count,
            'analyst_only_count': len(analyst_only),
            'analyst_initiated_ratio': analyst_initiated_count / total_ai_exchanges if total_ai_exchanges > 0 else 0.0,
            'management_pivot_ratio': management_pivot_count / total_ai_exchanges if total_ai_exchanges > 0 else 0.0,
            'ai_initiation_score': ai_initiation_score
        })
    
    return pd.DataFrame(results)


def compute_all_initiation_metrics(
    sentences_df: pd.DataFrame,
    output_dir: str = "outputs/features"
) -> pd.DataFrame:
    """
    Full pipeline to compute initiation scores.
    
    Args:
        sentences_df: Sentence-level data with predictions
        output_dir: Output directory
        
    Returns:
        DataFrame with initiation scores
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting Q&A exchanges...")
    exchanges = extract_qa_exchanges(sentences_df)
    print(f"Found {len(exchanges)} Q&A exchanges")
    
    print("Computing initiation scores...")
    scores_df = compute_initiation_scores(exchanges)
    
    # Save
    scores_df.to_parquet(f"{output_dir}/initiation_scores.parquet", index=False)
    
    print(f"\n=== Initiation Score Summary ===")
    print(f"Documents with AI exchanges: {(scores_df['total_ai_exchanges'] > 0).sum()}")
    print(f"Avg AI exchanges per doc: {scores_df['total_ai_exchanges'].mean():.1f}")
    print(f"Avg analyst-initiated ratio: {scores_df['analyst_initiated_ratio'].mean():.3f}")
    print(f"Avg management-pivot ratio: {scores_df['management_pivot_ratio'].mean():.3f}")
    print(f"Avg AI initiation score: {scores_df['ai_initiation_score'].mean():.3f}")
    print("  (Higher = more management-driven)")
    
    return scores_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute AI initiation scores")
    parser.add_argument("--input", default="outputs/features/sentences_with_predictions.parquet")
    parser.add_argument("--output-dir", default="outputs/features")
    
    args = parser.parse_args()
    
    sentences_df = pd.read_parquet(args.input)
    compute_all_initiation_metrics(sentences_df, args.output_dir)
