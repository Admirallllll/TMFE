from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import FeatureConfig
from src.features import transfer_inference
from src.models.train_transfer_encoder import TransferEncoderBundle


class _DummyLogger:
    def info(self, *_args, **_kwargs):
        return None


def test_compute_transfer_features_outputs_expected_columns(monkeypatch):
    df = pd.DataFrame({"clean_transcript": ["alpha", "beta", "gamma"]})

    def _fake_predict_logits(texts, *, bundle, batch_size, max_len, logger):
        assert len(texts) == 3
        return np.array([-2.0, 0.0, 2.0], dtype="float64")

    monkeypatch.setattr(transfer_inference, "_predict_logits", _fake_predict_logits)

    bundle = TransferEncoderBundle(
        model_dir=Path("."),
        model_name="stub",
        threshold=0.5,
        device="cpu",
        max_len=64,
        model=None,
        tokenizer=None,
    )
    cfg = FeatureConfig(transfer_batch_size=8, transfer_max_len=64)
    res = transfer_inference.compute_transfer_features(df, encoder_bundle=bundle, cfg=cfg, logger=_DummyLogger())

    assert list(res.features.columns) == ["transfer_ai_prob", "transfer_ai_logit", "transfer_ai_confidence"]
    assert np.all((res.features["transfer_ai_prob"].to_numpy() >= 0.0) & (res.features["transfer_ai_prob"].to_numpy() <= 1.0))
    assert np.allclose(res.features["transfer_ai_logit"].to_numpy(), np.array([-2.0, 0.0, 2.0]))
    assert np.all((res.features["transfer_ai_confidence"].to_numpy() >= 0.0) & (res.features["transfer_ai_confidence"].to_numpy() <= 1.0))
