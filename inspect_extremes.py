import pandas as pd
import numpy as np

def inspect_extremes():
    print("Loading metrics...")
    try:
        df = pd.read_parquet("outputs/features/document_metrics.parquet")
    except FileNotFoundError:
        print("Error: content file not found.")
        return

    # --- Document Level ---
    print("\n" + "="*50)
    print("DOCUMENT EXTREMES (from quadrant_scatter_documents.png)")
    print("="*50)
    
    # 1. Top-Right (Highest Combined Intensity)
    # Heuristic: maximize sum of both ratios
    df['combined_ratio'] = df['speech_kw_ai_ratio'] + df['qa_kw_ai_ratio']
    top_right_doc = df.loc[df['combined_ratio'].idxmax()]
    
    print(f"\n[Top-Right] Most 'Aligned' Document (Max Combined Intensity):")
    print(f"  Doc ID: {top_right_doc['doc_id']}")
    print(f"  Speech AI Ratio: {top_right_doc['speech_kw_ai_ratio']:.4f}")
    print(f"  Q&A AI Ratio:    {top_right_doc['qa_kw_ai_ratio']:.4f}")

    # 2. Top-Left (High Q&A, Low Speech)
    # Heuristic: Max Q&A among those with Speech < Mean
    speech_mean = df['speech_kw_ai_ratio'].mean()
    passive_docs = df[df['speech_kw_ai_ratio'] < speech_mean]
    if not passive_docs.empty:
        top_left_doc = passive_docs.loc[passive_docs['qa_kw_ai_ratio'].idxmax()]
        print(f"\n[Top-Left] Most 'Passive' Document (High Q&A, Low Speech):")
        print(f"  Doc ID: {top_left_doc['doc_id']}")
        print(f"  Speech AI Ratio: {top_left_doc['speech_kw_ai_ratio']:.4f}")
        print(f"  Q&A AI Ratio:    {top_left_doc['qa_kw_ai_ratio']:.4f}")
    
    # 3. Right-Most (Max Speech)
    right_most_doc = df.loc[df['speech_kw_ai_ratio'].idxmax()]
    print(f"\n[Right-Most] Most 'Self-Promoting' Document (Max Speech):")
    print(f"  Doc ID: {right_most_doc['doc_id']}")
    print(f"  Speech AI Ratio: {right_most_doc['speech_kw_ai_ratio']:.4f}")
    print(f"  Q&A AI Ratio:    {right_most_doc['qa_kw_ai_ratio']:.4f}")

    # --- Company Level ---
    print("\n" + "="*50)
    print("COMPANY EXTREMES (from quadrant_scatter_companies.png)")
    print("="*50)

    # Aggregate to company level
    df['ticker'] = df['doc_id'].apply(lambda x: str(x).rsplit('_', 1)[0] if '_' in str(x) else x)
    comp_df = df.groupby('ticker').agg({
        'speech_kw_ai_ratio': 'mean',
        'qa_kw_ai_ratio': 'mean',
        'doc_id': 'count'
    }).reset_index()
    
    # 1. Top-Right (Company)
    comp_df['combined_ratio'] = comp_df['speech_kw_ai_ratio'] + comp_df['qa_kw_ai_ratio']
    top_right_comp = comp_df.loc[comp_df['combined_ratio'].idxmax()]
    
    print(f"\n[Top-Right] Most 'Aligned' Company:")
    print(f"  Ticker: {top_right_comp['ticker']}")
    print(f"  Avg Speech Ratio: {top_right_comp['speech_kw_ai_ratio']:.4f}")
    print(f"  Avg Q&A Ratio:    {top_right_comp['qa_kw_ai_ratio']:.4f}")
    print(f"  Num Calls:        {top_right_comp['doc_id']}")

    # 2. Top-Left (Company)
    comp_speech_mean = comp_df['speech_kw_ai_ratio'].mean()
    comp_passive = comp_df[comp_df['speech_kw_ai_ratio'] < comp_speech_mean]
    if not comp_passive.empty:
        top_left_comp = comp_passive.loc[comp_passive['qa_kw_ai_ratio'].idxmax()]
        print(f"\n[Top-Left] Most 'Passive' Company:")
        print(f"  Ticker: {top_left_comp['ticker']}")
        print(f"  Avg Speech Ratio: {top_left_comp['speech_kw_ai_ratio']:.4f}")
        print(f"  Avg Q&A Ratio:    {top_left_comp['qa_kw_ai_ratio']:.4f}")
    
    # 3. Right-Most (Company)
    right_most_comp = comp_df.loc[comp_df['speech_kw_ai_ratio'].idxmax()]
    print(f"\n[Right-Most] Most 'Self-Promoting' Company:")
    print(f"  Ticker: {right_most_comp['ticker']}")
    print(f"  Avg Speech Ratio: {right_most_comp['speech_kw_ai_ratio']:.4f}")
    print(f"  Avg Q&A Ratio:    {right_most_comp['qa_kw_ai_ratio']:.4f}")

if __name__ == "__main__":
    inspect_extremes()
