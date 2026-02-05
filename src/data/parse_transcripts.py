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
