import pandas as pd

def inspect_documents():
    df = pd.read_parquet("outputs/features/document_metrics.parquet")
    
    with open("extremes_details.txt", "w", encoding="utf-8") as f:
        # 1. Top-Right (Max combined)
        df['combined'] = df['speech_kw_ai_ratio'] + df['qa_kw_ai_ratio']
        tr_doc = df.loc[df['combined'].idxmax()]
        f.write(f"Top-Right Document: {tr_doc['doc_id']}\n")
        f.write(f"  Speech Ratio: {tr_doc['speech_kw_ai_ratio']:.4f}\n")
        f.write(f"  Q&A Ratio:    {tr_doc['qa_kw_ai_ratio']:.4f}\n\n")
        
        # 2. Top-Left (High Q&A, Low Speech < 0.01)
        # Using strict criteria to find the one in the top-left corner
        # Sort by Q&A descending, filter by Speech < 0.01 (near zero)
        tl_candidates = df[df['speech_kw_ai_ratio'] < 0.005].sort_values('qa_kw_ai_ratio', ascending=False)
        if not tl_candidates.empty:
            tl_doc = tl_candidates.iloc[0]
            f.write(f"Top-Left (Passive) Document: {tl_doc['doc_id']}\n")
            f.write(f"  Speech Ratio: {tl_doc['speech_kw_ai_ratio']:.4f}\n")
            f.write(f"  Q&A Ratio:    {tl_doc['qa_kw_ai_ratio']:.4f}\n\n")
        
        # 3. Right-Most (Max Speech)
        rm_doc = df.loc[df['speech_kw_ai_ratio'].idxmax()]
        f.write(f"Right-Most Document: {rm_doc['doc_id']}\n")
        f.write(f"  Speech Ratio: {rm_doc['speech_kw_ai_ratio']:.4f}\n")
        f.write(f"  Q&A Ratio:    {rm_doc['qa_kw_ai_ratio']:.4f}\n")

if __name__ == "__main__":
    inspect_documents()
