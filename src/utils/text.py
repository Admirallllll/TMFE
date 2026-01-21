from __future__ import annotations

import re
import unicodedata


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_WS_RE = re.compile(r"\s+")


def clean_text_basic(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize_simple(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def count_tokens(text: str) -> int:
    return len(tokenize_simple(text))

