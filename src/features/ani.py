from __future__ import annotations

import pandas as pd

from src.features.ai_dictionary import ai_term_patterns


AI_GROUPS: dict[str, tuple[str, ...]] = {
    "ai_core": ("artificial_intelligence", "ai"),
    "ml": ("machine_learning", "ml", "deep_learning", "neural_network"),
    "llm": ("large_language_model", "llm"),
    "genai": (
        "generative_ai",
        "genai",
        "large_language_model",
        "llm",
        "foundation_model",
        "transformer",
        "chatgpt",
        "copilot",
        "prompt",
        "rag",
        "retrieval_augmented",
    ),
}


def compute_ani_features(
    df: pd.DataFrame,
    *,
    text_col: str = "clean_transcript",
    token_col: str = "n_tokens",
    logger=None,
) -> pd.DataFrame:
    patterns = ai_term_patterns()
    s = df[text_col].fillna("")

    counts: dict[str, pd.Series] = {}
    for key, pat in patterns.items():
        counts[key] = s.str.count(pat).astype("int64")

    out = pd.DataFrame({f"ani_count_{k}": v for k, v in counts.items()})
    out["ani_count_total"] = out.sum(axis=1).astype("int64")

    denom = df[token_col].fillna(0).astype("int64").clip(lower=1).astype(float)
    out["ani_kw_per1k"] = (out["ani_count_total"] / denom) * 1000.0

    for group, keys in AI_GROUPS.items():
        group_count = sum(out[f"ani_count_{k}"] for k in keys)
        out[f"ani_{group}_per1k"] = (group_count / denom) * 1000.0

    out["ani_any"] = (out["ani_count_total"] > 0).astype("int8")

    if logger is not None:
        logger.info(
            "ANI features computed: "
            + ", ".join(
                [
                    f"mean ani_kw_per1k={out['ani_kw_per1k'].mean():.3f}",
                    f"share any={out['ani_any'].mean():.3f}",
                ]
            )
        )

    return out
