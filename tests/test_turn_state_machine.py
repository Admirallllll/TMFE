from __future__ import annotations

from src.data.parse_transcripts import extract_turns


def test_turn_state_machine_question_answer_grouping():
    qa = (
        "ANALYST: What about AI?\n"
        "CEO: We invest.\n"
        "CFO: And cost savings.\n"
        "ANALYST: Follow-up.\n"
        "OPERATOR: Next question.\n"
        "CEO: More AI.\n"
    )
    turns, _diag = extract_turns(qa, roster={})
    question_turns = [t for t in turns if t.turn_type == "question"]
    assert [t.question_id for t in question_turns] == [1, 2]
    answer_turns = [t for t in turns if t.turn_type == "answer"]
    assert [t.answer_group_id for t in answer_turns] == [1, 1, 2]

