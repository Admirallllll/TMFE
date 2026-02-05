from __future__ import annotations

from src.data.parse_transcripts import split_prepared_and_qa


def test_split_prepared_and_qa_basic():
    txt = (
        "Prepared Remarks\n"
        "We had a strong quarter.\n"
        "Question-and-Answer Session\n"
        "OPERATOR: Our first question...\n"
    )
    prepared, qa, diag = split_prepared_and_qa(txt)
    assert diag["qa_header_matched"] is True
    assert diag["qa_start_idx"] >= 0
    assert diag["prepared_len"] == len(prepared)
    assert diag["qa_len"] == len(qa)
    assert "Question-and-Answer Session" not in prepared
    assert "OPERATOR" in qa


def test_split_prepared_and_qa_no_header():
    txt = "Prepared Remarks\nWe had a strong quarter.\nNo Q&A here."
    prepared, qa, diag = split_prepared_and_qa(txt)
    assert diag["qa_header_matched"] is False
    assert diag["qa_start_idx"] == -1
    assert qa == ""
    assert prepared == txt
