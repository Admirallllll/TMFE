from __future__ import annotations

import json
from collections import Counter

import pandas as pd


POS_STRONG: set[str] = {
    "openai",
    "chatgpt",
    "gpt",
    "gpt-4",
    "gpt4",
    "gpt-4o",
    "llama",
    "llama2",
    "llama3",
    "claude",
    "gemini",
    "mistral",
    "qwen",
    "deepseek",
    "generativeai",
    "genai",
    "llm",
    "large-language-model",
    "foundation-model",
    "ai-agent",
    "prompt-engineering",
    "rag",
    "retrieval-augmented-generation",
    "multimodal-ai",
    "copilot",
}

POS_WEAK: set[str] = {
    "machine-learning",
    "deep-learning",
    "neural-network",
    "nlp",
    "computer-vision",
    "model-training",
    "fine-tuning",
    "inference",
    "ai-safety",
    "responsible-ai",
    "synthetic-data",
    "vector-database",
}

NEG_STRONG: set[str] = {
    "missiles",
    "weapon",
    "warfare",
    "military",
    "celebrity",
    "movie",
    "music",
    "fashion",
    "travel",
    "cooking",
    "football",
    "basketball",
    "baseball",
}

NEG_WEAK: set[str] = {
    "robotics",
    "video",
    "gaming",
    "smartphone",
    "gadget",
    "blockchain",
    "crypto",
    "web3",
    "iot",
    "drone",
}


def normalize_tag(value: str) -> str:
    tag = str(value).strip().lower()
    for ch in ["_", "/", "\\", ":", ";", ",", ".", "(", ")", "[", "]", "{", "}", "'"]:
        tag = tag.replace(ch, " ")
    return "-".join(part for part in tag.split() if part)


def _parse_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = [str(v) for v in value]
    elif isinstance(value, (tuple, set)):
        raw = [str(v) for v in value]
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                raw = [str(v) for v in arr] if isinstance(arr, list) else [s]
            except json.JSONDecodeError:
                raw = [x.strip() for x in s.strip("[]").split(",") if x.strip()]
        else:
            for sep in ("|", ",", ";"):
                if sep in s:
                    raw = [x.strip() for x in s.split(sep) if x.strip()]
                    break
            else:
                raw = [s]
    else:
        raw = [str(value)]

    return sorted({normalize_tag(v) for v in raw if str(v).strip()})


def _tag_bucket(tag: str) -> str:
    if tag in POS_STRONG:
        return "POS_STRONG"
    if tag in POS_WEAK:
        return "POS_WEAK"
    if tag in NEG_STRONG:
        return "NEG_STRONG"
    if tag in NEG_WEAK:
        return "NEG_WEAK"
    return "OTHER"


def label_from_tags(tags: list[str], *, label_margin: int = 3) -> dict[str, object]:
    norm_tags = [normalize_tag(t) for t in tags if str(t).strip()]
    uniq = sorted(set(norm_tags))

    pos_strong = sorted([t for t in uniq if t in POS_STRONG])
    pos_weak = sorted([t for t in uniq if t in POS_WEAK])
    neg_strong = sorted([t for t in uniq if t in NEG_STRONG])
    neg_weak = sorted([t for t in uniq if t in NEG_WEAK])

    score = (3 * len(pos_strong)) + len(pos_weak) - (3 * len(neg_strong)) - len(neg_weak)
    label: int | None
    reason: str

    if pos_strong and (not neg_strong) and (score >= int(label_margin)):
        label = 1
        reason = "high_precision_positive"
    elif neg_strong and (not pos_strong) and (score <= -int(label_margin)):
        label = 0
        reason = "high_precision_negative"
    else:
        label = None
        if pos_strong and neg_strong:
            reason = "conflicting_strong_tags"
        elif score > 0:
            reason = "weak_positive_or_low_margin"
        elif score < 0:
            reason = "weak_negative_or_low_margin"
        else:
            reason = "no_signal"

    return {
        "label": label,
        "score": int(score),
        "reason": reason,
        "pos_strong_hits": pos_strong,
        "pos_weak_hits": pos_weak,
        "neg_strong_hits": neg_strong,
        "neg_weak_hits": neg_weak,
        "tags_norm": uniq,
    }


def build_transfer_labels(
    df: pd.DataFrame,
    *,
    min_tag_freq: int,
    label_margin: int,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "source_text" not in df.columns:
        raise ValueError("Expected source_text column in AI media dataframe")
    if "tags_norm" not in df.columns and "tags_raw" not in df.columns:
        raise ValueError("Expected tags_norm or tags_raw column in AI media dataframe")

    work = df.copy()
    if "tags_norm" not in work.columns:
        work["tags_norm"] = work["tags_raw"].map(_parse_tags)
    else:
        work["tags_norm"] = work["tags_norm"].map(_parse_tags)

    counts = Counter(tag for tags in work["tags_norm"] for tag in tags)
    always_keep = set(POS_STRONG) | set(NEG_STRONG)
    eligible = {tag for tag, n in counts.items() if n >= int(min_tag_freq)} | always_keep

    def _apply(tags: list[str]) -> dict[str, object]:
        tags_kept = [t for t in tags if t in eligible]
        return label_from_tags(tags_kept, label_margin=label_margin)

    label_payload = work["tags_norm"].map(_apply)
    meta = pd.DataFrame(label_payload.tolist())
    out = pd.concat([work.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    out["label_transfer"] = out["label"].astype("Int64")

    audit_rows: list[dict[str, object]] = []
    for tag, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        audit_rows.append(
            {
                "tag": tag,
                "freq": int(n),
                "bucket": _tag_bucket(tag),
                "eligible": bool(tag in eligible),
            }
        )
    audit = pd.DataFrame(audit_rows)

    n_total = len(out)
    n_pos = int((out["label_transfer"] == 1).sum())
    n_neg = int((out["label_transfer"] == 0).sum())
    n_review = int(out["label_transfer"].isna().sum())
    logger.info(
        "Transfer labels built: "
        + f"total={n_total:,}, pos={n_pos:,}, neg={n_neg:,}, review={n_review:,}, min_tag_freq={min_tag_freq}"
    )

    return out, audit
