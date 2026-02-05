from __future__ import annotations

import re
from typing import Iterable


_QA_HEADERS_NORM: tuple[str, ...] = (
    "question and answer session",
    "questions and answers",
    "question and answer",
    "q a",
)

_ROSTER_HEADERS: tuple[str, ...] = (
    "corporate participants",
    "company participants",
    "participants",
    "analysts",
)

_SUFFIX_TOKENS: set[str] = {"cfa", "cpa", "phd", "jr", "sr", "ii", "iii", "iv"}


class Turn:
    def __init__(
        self,
        *,
        turn_id: int,
        speaker_name: str,
        speaker_role: str,
        speaker_raw_header: str,
        char_start: int,
        char_end: int,
        turn_text_raw: str,
        turn_text_clean: str,
        turn_type: str,
        question_id: int | None,
        answer_group_id: int | None,
        roster_matched: bool,
    ) -> None:
        self.turn_id = turn_id
        self.speaker_name = speaker_name
        self.speaker_role = speaker_role
        self.speaker_raw_header = speaker_raw_header
        self.char_start = char_start
        self.char_end = char_end
        self.turn_text_raw = turn_text_raw
        self.turn_text_clean = turn_text_clean
        self.turn_type = turn_type
        self.question_id = question_id
        self.answer_group_id = answer_group_id
        self.roster_matched = roster_matched


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", " ", name.lower())
    cleaned = re.sub(r"[^a-z\\s]", " ", cleaned)
    tokens = [t for t in cleaned.split() if t and t not in _SUFFIX_TOKENS]
    if len(tokens) >= 2:
        first, last = tokens[0], tokens[-1]
        return f"{first} {last}".strip()
    return " ".join(tokens).strip()


def _find_qa_header(text: str) -> tuple[int, int, str | None]:
    best_start = -1
    best_end = -1
    best_pattern: str | None = None
    for match in re.finditer(r"(?m)^.*$", text):
        line = match.group(0).strip()
        if not line:
            continue
        norm = re.sub(r"[^a-z]+", " ", line.lower()).strip()
        for header in _QA_HEADERS_NORM:
            if norm.startswith(header):
                best_start = match.start()
                best_end = match.end()
                best_pattern = header
                return best_start, best_end, best_pattern
    return best_start, best_end, best_pattern


def split_prepared_and_qa(transcript_raw: str) -> tuple[str, str, dict[str, object]]:
    if not transcript_raw:
        diag = {
            "qa_header_matched": False,
            "pattern_used": None,
            "qa_start_idx": -1,
            "prepared_len": 0,
            "qa_len": 0,
        }
        return "", "", diag

    start, end, pattern = _find_qa_header(transcript_raw)
    if start == -1:
        diag = {
            "qa_header_matched": False,
            "pattern_used": None,
            "qa_start_idx": -1,
            "prepared_len": len(transcript_raw),
            "qa_len": 0,
        }
        return transcript_raw, "", diag

    newline_idx = transcript_raw.find("\n", end)
    qa_start_idx = newline_idx + 1 if newline_idx != -1 else end
    prepared = transcript_raw[:start].rstrip()
    qa = transcript_raw[qa_start_idx:].lstrip()

    diag = {
        "qa_header_matched": True,
        "pattern_used": pattern,
        "qa_start_idx": qa_start_idx,
        "prepared_len": len(prepared),
        "qa_len": len(qa),
    }
    return prepared, qa, diag


def _iter_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        yield line.strip()


def extract_rosters(transcript_raw: str) -> dict[str, str]:
    roster: dict[str, str] = {}
    current_section: str | None = None
    for line in _iter_lines(transcript_raw):
        if not line:
            current_section = None
            continue

        header = line.lower()
        if any(h in header for h in _ROSTER_HEADERS):
            if "analyst" in header:
                current_section = "analyst"
            elif "participant" in header:
                current_section = "management"
            else:
                current_section = None
            continue

        if current_section is None:
            continue

        if line.isupper() and len(line.split()) >= 2:
            current_section = None
            continue

        name_part = re.split(r"\s+[-–—]\s+|,", line, maxsplit=1)[0].strip()
        if not name_part:
            continue
        norm = normalize_name(name_part)
        if not norm:
            continue
        roster[norm] = current_section
    return roster


def _guess_role(header: str, roster: dict[str, str]) -> tuple[str, str, bool]:
    raw = header.strip()
    norm = normalize_name(raw)
    if norm and norm in roster:
        return norm, roster[norm], True

    lower = raw.lower()
    if "operator" in lower:
        return norm or raw, "operator", False
    if "analyst" in lower:
        return norm or raw, "analyst", False
    if raw.strip().upper() == "ANALYST":
        return norm or raw, "analyst", False
    if re.fullmatch(r"[A-Z]{2,6}", raw.strip()):
        return norm or raw, "management", False
    if raw.strip() in {"CEO", "CFO", "COO", "CTO", "CMO", "CIO"}:
        return norm or raw, "management", False
    if any(title in lower for title in ["ceo", "cfo", "chief", "president", "vp", "vice president", "cto", "coo"]):
        return norm or raw, "management", False
    return norm or raw, "other", False


_SPEAKER_RE = re.compile(r"^(?P<header>[A-Z][^:]{1,120}):\s*(?P<text>.*)$", re.M)


def extract_turns(qa_text: str, roster: dict[str, str]) -> tuple[list[Turn], dict[str, object]]:
    turns: list[Turn] = []
    question_id = 0
    answer_group_id: int | None = None
    current_header = None
    current_role = None
    current_start = 0
    current_text = []

    def _flush(end_idx: int) -> None:
        nonlocal question_id, answer_group_id, current_header, current_role, current_start, current_text
        if current_header is None:
            return
        raw_text = "\n".join(current_text).strip()
        if not raw_text:
            current_header = None
            current_text = []
            return
        speaker_name, speaker_role, roster_matched = _guess_role(current_header, roster)
        if speaker_role == "analyst":
            question_id += 1
            answer_group_id = question_id
            turn_type = "question"
            q_id = question_id
            a_id = None
        elif speaker_role == "management":
            if answer_group_id is None:
                question_id += 1
                answer_group_id = question_id
            turn_type = "answer"
            q_id = question_id if question_id > 0 else None
            a_id = answer_group_id
        else:
            turn_type = "other"
            q_id = question_id if question_id > 0 else None
            a_id = answer_group_id

        turns.append(
            Turn(
                turn_id=len(turns) + 1,
                speaker_name=speaker_name,
                speaker_role=speaker_role,
                speaker_raw_header=current_header,
                char_start=current_start,
                char_end=end_idx,
                turn_text_raw=raw_text,
                turn_text_clean=" ".join(raw_text.split()),
                turn_type=turn_type,
                question_id=q_id,
                answer_group_id=a_id,
                roster_matched=roster_matched,
            )
        )
        current_header = None
        current_text = []

    for match in _SPEAKER_RE.finditer(qa_text):
        header = match.group("header").strip()
        start = match.start()
        if current_header is not None:
            _flush(start)
        current_header = header
        current_role = None
        current_start = start
        text = match.group("text")
        current_text = [text] if text else []

    if current_header is not None:
        _flush(len(qa_text))

    diag = {
        "turn_count": len(turns),
    }
    return turns, diag
