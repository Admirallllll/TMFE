from pathlib import Path

import pandas as pd

from scripts.export_annotation_samples import export_ai_sentence_audit


def test_export_ai_sentence_audit_deduplicates_display_level_duplicates(tmp_path: Path):
    src = pd.DataFrame(
        [
            {"doc_id": "D1", "section": "qa", "text": "Thank you.", "kw_is_ai": False},
            {"doc_id": "D1", "section": "qa", "text": "Thank you.", "kw_is_ai": False},
            {"doc_id": "D1", "section": "qa", "text": "Different text", "kw_is_ai": False},
            {"doc_id": "D2", "section": "speech", "text": "AI model deployment", "kw_is_ai": True},
            {"doc_id": "D2", "section": "speech", "text": "AI model deployment", "kw_is_ai": True},
            {"doc_id": "D3", "section": "speech", "text": "Another AI sentence", "kw_is_ai": True},
        ]
    )
    parquet_path = tmp_path / "sentences.parquet"
    src.to_parquet(parquet_path, index=False)

    out_path = export_ai_sentence_audit(
        sentences_kw_path=parquet_path,
        output_dir=tmp_path,
        pos_n=2,
        neg_n=2,
        seed=42,
    )

    exported = pd.read_csv(out_path)

    assert exported.duplicated(subset=["doc_id", "section", "text"]).sum() == 0
