from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.text import tokenize_simple


LM_MASTERDICT_DRIVE_ID = "1cfg_w3USlRFS97wo7XQmYnuzhpmzboAY"
LM_MASTERDICT_URL = f"https://drive.google.com/uc?export=download&id={LM_MASTERDICT_DRIVE_ID}"


@dataclass(frozen=True)
class LMWords:
    positive: set[str]
    negative: set[str]
    uncertainty: set[str]


def _download_from_google_drive(url: str, dest_path: Path) -> None:
    import re

    import requests

    session = requests.Session()
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()

    token = None
    if "text/html" in response.headers.get("Content-Type", ""):
        m = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
        if m:
            token = m.group(1)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v

    if token:
        response = session.get(url + "&confirm=" + token, stream=True, timeout=60)
        response.raise_for_status()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)


def _ensure_lm_cached(external_dir: Path, *, logger) -> Path:
    lm_dir = external_dir / "lm_dict"
    lm_csv = lm_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
    if lm_csv.exists():
        return lm_csv

    logger.info("Downloading Loughran–McDonald master dictionary (CSV) from Notre Dame (Google Drive)")
    _download_from_google_drive(LM_MASTERDICT_URL, lm_csv)
    return lm_csv


def load_lm_word_sets(external_dir: Path, *, logger) -> LMWords:
    lm_csv = _ensure_lm_cached(external_dir, logger=logger)
    master = pd.read_csv(lm_csv)

    col_map = {c.lower(): c for c in master.columns}
    word_col = col_map.get("word")
    pos_col = col_map.get("positive")
    neg_col = col_map.get("negative")
    unc_col = col_map.get("uncertainty")
    missing = [c for c in (word_col, pos_col, neg_col, unc_col) if c is None]
    if missing:
        raise ValueError(f"LM master dictionary missing expected columns. Found: {list(master.columns)}")

    words = master[word_col].astype(str).str.upper()
    positive = set(words.loc[master[pos_col].fillna(0).astype(int) > 0])
    negative = set(words.loc[master[neg_col].fillna(0).astype(int) > 0])
    uncertainty = set(words.loc[master[unc_col].fillna(0).astype(int) > 0])

    logger.info(
        "Loaded LM word lists: "
        + ", ".join(
            [
                f"positive={len(positive):,}",
                f"negative={len(negative):,}",
                f"uncertainty={len(uncertainty):,}",
            ]
        )
    )
    return LMWords(positive=positive, negative=negative, uncertainty=uncertainty)


def _build_lexicon(words: LMWords) -> dict[str, int]:
    lex: dict[str, int] = {}
    for w in words.positive:
        lex[w] = lex.get(w, 0) | 1
    for w in words.negative:
        lex[w] = lex.get(w, 0) | 2
    for w in words.uncertainty:
        lex[w] = lex.get(w, 0) | 4
    return lex


def compute_lm_features(
    df: pd.DataFrame,
    *,
    text_col: str = "clean_transcript",
    token_col: str = "n_tokens",
    external_dir: Path,
    logger,
) -> pd.DataFrame:
    try:
        words = load_lm_word_sets(external_dir, logger=logger)
        lex = _build_lexicon(words)
    except Exception as e:
        logger.info(f"LM dictionary load failed ({e}); emitting zero LM features")
        n = len(df)
        denom = df[token_col].fillna(0).astype("int64").clip(lower=1).astype(float)
        out = pd.DataFrame(
            {
                "lm_pos_count": pd.Series([0] * n, dtype="int64"),
                "lm_neg_count": pd.Series([0] * n, dtype="int64"),
                "lm_unc_count": pd.Series([0] * n, dtype="int64"),
            }
        )
        out["lm_pos_per1k"] = (out["lm_pos_count"] / denom) * 1000.0
        out["lm_neg_per1k"] = (out["lm_neg_count"] / denom) * 1000.0
        out["lm_unc_per1k"] = (out["lm_unc_count"] / denom) * 1000.0
        out["lm_net_tone_per1k"] = 0.0
        return out

    denom = df[token_col].fillna(0).astype("int64").clip(lower=1).astype(float)

    pos_counts = []
    neg_counts = []
    unc_counts = []

    texts = df[text_col].fillna("").tolist()
    for text in texts:
        pos = 0
        neg = 0
        unc = 0
        for t in tokenize_simple(text):
            flags = lex.get(t.upper())
            if flags is None:
                continue
            if flags & 1:
                pos += 1
            if flags & 2:
                neg += 1
            if flags & 4:
                unc += 1
        pos_counts.append(pos)
        neg_counts.append(neg)
        unc_counts.append(unc)

    out = pd.DataFrame(
        {
            "lm_pos_count": pd.Series(pos_counts, dtype="int64"),
            "lm_neg_count": pd.Series(neg_counts, dtype="int64"),
            "lm_unc_count": pd.Series(unc_counts, dtype="int64"),
        }
    )

    out["lm_pos_per1k"] = (out["lm_pos_count"] / denom) * 1000.0
    out["lm_neg_per1k"] = (out["lm_neg_count"] / denom) * 1000.0
    out["lm_unc_per1k"] = (out["lm_unc_count"] / denom) * 1000.0
    out["lm_net_tone_per1k"] = ((out["lm_pos_count"] - out["lm_neg_count"]) / denom) * 1000.0

    logger.info(
        "LM features computed: "
        + ", ".join(
            [
                f"mean pos_per1k={out['lm_pos_per1k'].mean():.3f}",
                f"mean neg_per1k={out['lm_neg_per1k'].mean():.3f}",
                f"mean unc_per1k={out['lm_unc_per1k'].mean():.3f}",
            ]
        )
    )
    return out
