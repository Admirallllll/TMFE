from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_kaggle_ai_media import load_ai_media_dataset


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


def test_load_ai_media_prefers_local_file(tmp_path):
    local_csv = tmp_path / "ai_media_dataset_20250911.csv"
    pd.DataFrame(
        {
            "content": ["OpenAI released new model", "Sports update"],
            "tags": ["OpenAI,GenerativeAI", "football,video"],
        }
    ).to_csv(local_csv, index=False)

    df = load_ai_media_dataset(_DummyLogger(), local_candidates=(local_csv,))
    assert len(df) == 2
    assert set(df.columns) >= {"source_text", "tags_raw", "tags_norm"}
    assert "openai" in df.iloc[0]["tags_norm"]


def test_load_ai_media_without_local_requires_kaggle_credentials(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "kagglehub":
            raise ModuleNotFoundError("No module named kagglehub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError) as exc:
        load_ai_media_dataset(_DummyLogger(), local_candidates=(tmp_path / "missing.csv",))
    assert "Kaggle credentials" in str(exc.value)
