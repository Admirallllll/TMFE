from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import FeatureConfig
from src.models.train_transfer_encoder import TransferEncoderBundle


@dataclass(frozen=True)
class TransferFeatureResult:
    features: pd.DataFrame
    threshold: float
    model_name: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_runtime(bundle: TransferEncoderBundle):
    model = bundle.model
    tokenizer = bundle.tokenizer
    if model is not None and tokenizer is not None:
        return model, tokenizer

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(str(bundle.model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(bundle.model_dir))
    return model, tokenizer


def _predict_logits(
    texts: list[str],
    *,
    bundle: TransferEncoderBundle,
    batch_size: int,
    max_len: int,
    logger,
) -> np.ndarray:
    import torch

    model, tokenizer = _load_runtime(bundle)
    resolved_device = bundle.device if bundle.device != "auto" else "cpu"
    if resolved_device.startswith("cuda") and (not torch.cuda.is_available()):
        resolved_device = "cpu"
    if resolved_device == "mps":
        has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if not has_mps:
            resolved_device = "cpu"
    device = torch.device(resolved_device)
    model = model.to(device)
    model.eval()

    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), int(batch_size)):
            batch = texts[i : i + int(batch_size)]
            tokens = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=int(max_len),
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(**tokens).logits.view(-1).detach().cpu().numpy().astype("float64")
            all_logits.append(logits)
    out = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,), dtype="float64")
    logger.info(f"Transfer inference complete: docs={len(texts):,}, batch_size={batch_size}, device={resolved_device}")
    return out


def compute_transfer_features(
    panel_df: pd.DataFrame,
    *,
    encoder_bundle: TransferEncoderBundle,
    cfg: FeatureConfig,
    logger,
) -> TransferFeatureResult:
    if "clean_transcript" not in panel_df.columns:
        raise ValueError("panel_df must contain clean_transcript column")

    texts = panel_df["clean_transcript"].fillna("").astype(str).tolist()
    logits = _predict_logits(
        texts,
        bundle=encoder_bundle,
        batch_size=int(cfg.transfer_batch_size),
        max_len=int(cfg.transfer_max_len),
        logger=logger,
    )
    probs = _sigmoid(logits)
    conf = 2.0 * np.abs(probs - 0.5)

    feats = pd.DataFrame(
        {
            "transfer_ai_prob": pd.Series(probs, dtype="float64"),
            "transfer_ai_logit": pd.Series(logits, dtype="float64"),
            "transfer_ai_confidence": pd.Series(conf, dtype="float64"),
        }
    )
    return TransferFeatureResult(features=feats, threshold=float(encoder_bundle.threshold), model_name=encoder_bundle.model_name)
