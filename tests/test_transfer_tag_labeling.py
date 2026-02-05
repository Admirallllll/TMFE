from __future__ import annotations

import pandas as pd

from src.features.transfer_tag_labeling import build_transfer_labels, label_from_tags, normalize_tag


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


def test_normalize_tag_basic():
    assert normalize_tag("  OpenAI  ") == "openai"
    assert normalize_tag("Large Language Model") == "large-language-model"
    assert normalize_tag("GenAI/LLM") == "genai-llm"


def test_label_from_tags_positive_and_negative():
    pos = label_from_tags(["openai", "llm", "video"], label_margin=3)
    assert pos["label"] == 1
    assert pos["reason"] == "high_precision_positive"
    assert pos["score"] >= 3

    neg = label_from_tags(["missiles", "robotics"], label_margin=3)
    assert neg["label"] == 0
    assert neg["reason"] == "high_precision_negative"
    assert neg["score"] <= -3


def test_label_from_tags_conflict_goes_to_review():
    conflict = label_from_tags(["openai", "missiles"], label_margin=3)
    assert conflict["label"] is None
    assert conflict["reason"] == "conflicting_strong_tags"


def test_label_from_tags_generic_ai_is_positive():
    pos = label_from_tags(["ai"], label_margin=3)
    assert pos["label"] == 1
    assert pos["reason"] == "high_precision_positive"


def test_build_transfer_labels_filters_low_frequency_tags():
    df = pd.DataFrame(
        {
            "source_text": ["a", "b", "c", "d"],
            "tags_norm": [
                ["openai", "llm"],
                ["openai", "video"],
                ["missiles", "video"],
                ["unique-rare-tag"],
            ],
        }
    )
    labeled, audit = build_transfer_labels(df, min_tag_freq=2, label_margin=3, logger=_DummyLogger())
    assert "label_transfer" in labeled.columns
    assert set(audit.columns) >= {"tag", "freq", "bucket", "eligible"}
    rare_row = audit.loc[audit["tag"] == "unique-rare-tag"].iloc[0]
    assert bool(rare_row["eligible"]) is False


def test_neg_strong_tags_not_filtered_by_min_freq():
    df = pd.DataFrame(
        {
            "source_text": ["only one"],
            "tags_norm": [["missiles"]],
        }
    )
    labeled, _ = build_transfer_labels(df, min_tag_freq=10, label_margin=3, logger=_DummyLogger())
    assert labeled.loc[0, "label_transfer"] == 0


def test_missing_tags_filtered_before_labeling():
    df = pd.DataFrame(
        {
            "source_text": ["x"],
            "tags_norm": [[None, "openai", "nan", ""]],
        }
    )
    labeled, audit = build_transfer_labels(df, min_tag_freq=1, label_margin=3, logger=_DummyLogger())
    tags = labeled.loc[0, "tags_norm"]
    assert "none" not in tags
    assert "nan" not in tags
    assert "" not in tags
    assert not (audit["tag"] == "nan").any()


def test_weak_tags_not_filtered_by_min_freq():
    df = pd.DataFrame(
        {
            "source_text": ["x"],
            "tags_norm": [["robotics"]],
        }
    )
    labeled, audit = build_transfer_labels(df, min_tag_freq=10, label_margin=3, logger=_DummyLogger())
    assert labeled.loc[0, "reason"] == "weak_negative_or_low_margin"
    robotics = audit.loc[audit["tag"] == "robotics"].iloc[0]
    assert bool(robotics["eligible"]) is True
