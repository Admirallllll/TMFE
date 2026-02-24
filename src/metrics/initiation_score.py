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


def extract_qa_exchanges(
    sentences_df: pd.DataFrame,
    doc_id_col: str = 'doc_id',
    section_col: str = 'section',
    role_col: str = 'role',
    turn_idx_col: str = 'turn_idx',
    text_col: str = 'text',
    kw_pred_col: str = 'kw_is_ai'
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

    # Ensure AI flag column exists
    if kw_pred_col not in qa_df.columns:
        qa_df[kw_pred_col] = False

    # Preserve sentence order within turns when available
    sort_cols = [doc_id_col, turn_idx_col]
    if "sentence_idx" in qa_df.columns:
        sort_cols.append("sentence_idx")
    qa_df = qa_df.sort_values(sort_cols)

    # Aggregate per turn (much faster than repeated filtering)
    agg_map = {
        text_col: lambda s: ' '.join(s.astype(str)),
        role_col: 'first',
        kw_pred_col: 'any'
    }
    if 'speaker' in qa_df.columns:
        agg_map['speaker'] = 'first'

    turns_df = (
        qa_df.groupby([doc_id_col, turn_idx_col], sort=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={
            text_col: 'text',
            role_col: 'role',
            kw_pred_col: 'is_ai'
        })
    )
    if 'speaker' not in turns_df.columns:
        turns_df['speaker'] = ''
    turns_df['role'] = turns_df['role'].fillna('unknown')

    # Match questions with answers per document
    n_docs = turns_df[doc_id_col].nunique()
    doc_groups = turns_df.groupby(doc_id_col, sort=False)
    for doc_id, doc_turns in tqdm(doc_groups, total=n_docs, desc="Q&A exchanges"):
        turns = doc_turns.sort_values(turn_idx_col).to_dict('records')

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
                    exchanges.append(QAExchange(
                        doc_id=doc_id,
                        exchange_idx=exchange_idx,
                        question_text=question['text'],
                        answer_text=answer_text,
                        questioner=question.get('speaker', ''),
                        answerer=answers[0].get('speaker', ''),
                        question_is_ai=question['is_ai'],
                        answer_is_ai=answer_is_ai
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
        'answer_is_ai': e.answer_is_ai
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
    output_dir: str = "outputs/features",
    figures_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Full pipeline to compute initiation scores.
    
    Args:
        sentences_df: Sentence-level data with keyword flags
        output_dir: Output directory
        
    Returns:
        DataFrame with initiation scores
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(output_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Extracting Q&A exchanges...")
    exchanges = extract_qa_exchanges(sentences_df)
    print(f"Found {len(exchanges)} Q&A exchanges")
    
    print("Computing initiation scores...")
    scores_df = compute_initiation_scores(exchanges)
    
    # Save
    scores_df.to_parquet(f"{output_dir}/initiation_scores.parquet", index=False)

    print(f"\n=== Initiation Score Summary ===")
    if len(scores_df) == 0 or 'total_ai_exchanges' not in scores_df.columns:
        print("No Q&A exchanges found in the data.")
    else:
        print(f"Documents with AI exchanges: {(scores_df['total_ai_exchanges'] > 0).sum()}")
        print(f"Avg AI exchanges per doc: {scores_df['total_ai_exchanges'].mean():.1f}")
        print(f"Avg analyst-initiated ratio: {scores_df['analyst_initiated_ratio'].mean():.3f}")
        print(f"Avg management-pivot ratio: {scores_df['management_pivot_ratio'].mean():.3f}")
        print(f"Avg AI initiation score: {scores_df['ai_initiation_score'].mean():.3f}")
        print("  (Higher = more management-driven)")

    # Visualizations
    if len(scores_df) > 0:
        try:
            plot_initiation_distributions(scores_df, figures_dir)
            plot_initiation_ratios(scores_df, figures_dir)
            plot_initiation_scatter(scores_df, figures_dir)
        except Exception as e:
            print(f"Warning: failed to generate initiation score plots: {e}")
    
    return scores_df


def plot_initiation_distributions(
    scores_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot distribution of AI initiation scores.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(scores_df["ai_initiation_score"], bins=30, kde=True, ax=ax, color="purple")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Distribution of AI Initiation Scores")
    ax.set_xlabel("AI Initiation Score (Higher = Management-Driven)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "ai_initiation_distribution.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved initiation score distribution plot to {output_path}")


def plot_initiation_ratios(
    scores_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot average initiation ratios.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    df = scores_df.copy()
    df = df[df["total_ai_exchanges"] > 0].copy()
    if len(df) == 0:
        print("No AI exchanges found for initiation ratio plot.")
        return

    if "analyst_only_count" not in df.columns:
        df["analyst_only_count"] = 0

    df["analyst_only_ratio"] = df["analyst_only_count"] / df["total_ai_exchanges"].replace(0, np.nan)

    ratios = {
        "Analyst Initiated": df["analyst_initiated_ratio"].mean(),
        "Management Pivot": df["management_pivot_ratio"].mean(),
        "Analyst Only": df["analyst_only_ratio"].mean(),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(ratios.keys()), list(ratios.values()), color=["steelblue", "darkorange", "gray"], alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Ratio")
    ax.set_title("Average AI Initiation Composition")
    ax.grid(True, axis="y", alpha=0.3)

    output_path = os.path.join(output_dir, "ai_initiation_ratios.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved initiation ratio plot to {output_path}")


def plot_initiation_scatter(
    scores_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot AI initiation score vs total AI exchanges.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        scores_df["total_ai_exchanges"],
        scores_df["ai_initiation_score"],
        alpha=0.6,
        color="teal",
        s=40
    )
    ax.set_xlabel("Total AI Exchanges (per document)")
    ax.set_ylabel("AI Initiation Score")
    ax.set_title("AI Initiation Score vs AI Exchange Volume")
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "ai_initiation_scatter.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved initiation scatter plot to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute AI initiation scores")
    parser.add_argument("--input", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/features")
    
    args = parser.parse_args()
    
    sentences_df = pd.read_parquet(args.input)
    compute_all_initiation_metrics(sentences_df, args.output_dir)
