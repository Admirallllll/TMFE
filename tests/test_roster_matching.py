from __future__ import annotations

from src.data.parse_transcripts import extract_rosters, normalize_name


def test_extract_rosters_basic():
    txt = (
        "CORPORATE PARTICIPANTS\n"
        "John A. Doe - Chief Executive Officer\n"
        "Jane B. Smith - Chief Financial Officer\n"
        "ANALYSTS\n"
        "Alice Johnson - Big Bank\n"
        "Bob Lee - Research Firm\n"
    )
    roster = extract_rosters(txt)
    assert roster.get("john doe") == "management"
    assert roster.get("jane smith") == "management"
    assert roster.get("alice johnson") == "analyst"
    assert roster.get("bob lee") == "analyst"


def test_normalize_name():
    assert normalize_name("John A. Doe") == "john doe"
    assert normalize_name("JANE SMITH, CFA") == "jane smith"


def test_extract_turns_uses_roster():
    from src.data.parse_transcripts import extract_turns

    qa = "John A. Doe: We are investing in AI."
    roster = {"john doe": "management"}
    turns, _diag = extract_turns(qa, roster=roster)
    assert len(turns) == 1
    assert turns[0].speaker_role == "management"
    assert turns[0].roster_matched is True
