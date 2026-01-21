from __future__ import annotations

import re


_DATACQTR_RE = re.compile(r"^(?P<year>\d{4})Q(?P<q>[1-4])$")


def parse_datacqtr(datacqtr: str) -> tuple[int, int]:
    if datacqtr is None:
        raise ValueError("datacqtr is None")
    match = _DATACQTR_RE.match(str(datacqtr).strip())
    if not match:
        raise ValueError(f"Invalid datacqtr: {datacqtr!r}")
    return int(match.group("year")), int(match.group("q"))


def quarter_index(year: int, quarter: int) -> int:
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Quarter must be 1-4, got {quarter}")
    return year * 4 + (quarter - 1)


def datacqtr_to_index(datacqtr: str) -> int:
    y, q = parse_datacqtr(datacqtr)
    return quarter_index(y, q)


def index_to_datacqtr(idx: int) -> str:
    year = idx // 4
    quarter = (idx % 4) + 1
    return f"{year}Q{quarter}"

