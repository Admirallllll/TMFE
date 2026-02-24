import pandas as pd

target_doc = 'NVDA_2023Q4'
try:
    df = pd.read_parquet('outputs/features/sentences_with_keywords.parquet')
    subset = df[df['doc_id'] == target_doc].copy()
    
    if 'turn_idx' in subset.columns:
        subset = subset.sort_values('turn_idx')
    
    output_path = 'nvda_transcript_view.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript for {target_doc}\n")
        f.write("="*80 + "\n")
        
        for idx, row in subset.iterrows():
            section = row['section'].upper()
            role = row['role']
            text = row['text']
            kw_pred = "YES" if row['kw_is_ai'] else "NO"
            
            f.write(f"[{section}] ({role}) {text}\n")
            f.write(f"   >> KW: {kw_pred}\n")
            f.write("-" * 80 + "\n")
            
    print(f"Saved to {output_path}")

except Exception as e:
    print(f"Error: {e}")
