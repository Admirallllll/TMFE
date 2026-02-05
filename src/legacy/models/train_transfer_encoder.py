from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import FeatureConfig, Paths
from src.utils.io import write_csv


@dataclass(frozen=True)
class TransferEncoderBundle:
    model_dir: Path
    model_name: str
    threshold: float
    device: str
    max_len: int
    model: object | None = None
    tokenizer: object | None = None


def _resolve_device(device: str) -> str:
    pref = (device or "").strip().lower()
    if pref in {"", "auto"}:
        pref = "cuda"

    try:
        import torch
    except Exception:
        return "cpu"

    if pref.startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return pref


def _artifact_paths(paths: Paths) -> tuple[Path, Path, Path]:
    model_dir = paths.transfer_model_dir / "encoder"
    config_path = model_dir / "transfer_config.json"
    metrics_path = paths.transfer_table_dir / "source_metrics.csv"
    return model_dir, config_path, metrics_path


def _has_saved_encoder(model_dir: Path, config_path: Path) -> bool:
    return model_dir.exists() and (model_dir / "config.json").exists() and config_path.exists()


def _build_dataset(encodings, labels, torch):
    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, enc, y):
            self.enc = enc
            self.y = y

        def __len__(self):
            return int(self.y.shape[0])

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = self.y[idx]
            return item

    return _Dataset(encodings, labels)


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (probs >= float(threshold)).astype(int)
    out: dict[str, float] = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        out["auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, probs))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def _select_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, num=19):
        f1 = float(f1_score(y_true, (probs >= thr).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def _load_bundle(
    model_dir: Path,
    config_path: Path,
    cfg: FeatureConfig,
    *,
    logger,
) -> TransferEncoderBundle:
    with open(config_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta.get("threshold", 0.5))
    model_name = str(meta.get("model_name", cfg.transfer_model_name))
    device = str(meta.get("device", _resolve_device(cfg.transfer_device)))
    max_len = int(meta.get("max_len", cfg.transfer_max_len))

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required to load transfer encoder") from e

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    logger.info(f"Loaded cached transfer encoder from {model_dir}")
    return TransferEncoderBundle(
        model_dir=model_dir,
        model_name=model_name,
        threshold=threshold,
        device=device,
        max_len=max_len,
        model=model,
        tokenizer=tokenizer,
    )


def train_or_load_transfer_encoder(
    df_labeled: pd.DataFrame,
    *,
    cfg: FeatureConfig,
    paths: Paths,
    logger,
    force_retrain: bool,
    max_train_samples: int | None,
    seed: int,
) -> TransferEncoderBundle:
    model_dir, config_path, metrics_path = _artifact_paths(paths)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths.transfer_table_dir.mkdir(parents=True, exist_ok=True)

    if (not force_retrain) and _has_saved_encoder(model_dir, config_path):
        return _load_bundle(model_dir, config_path, cfg, logger=logger)

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Transfer encoder training requires torch and transformers. Install requirements and retry."
        ) from e

    train_df = df_labeled.loc[df_labeled["label_transfer"].isin([0, 1]), ["source_text", "label_transfer"]].copy()
    train_df["label_transfer"] = train_df["label_transfer"].astype(int)
    if len(train_df) < 100:
        raise RuntimeError(f"Not enough labeled AI media samples for transfer training: {len(train_df)} rows")

    if max_train_samples is not None and len(train_df) > int(max_train_samples):
        train_df, _ = train_test_split(
            train_df,
            train_size=int(max_train_samples),
            random_state=seed,
            stratify=train_df["label_transfer"],
        )
        train_df = train_df.reset_index(drop=True)

    split_train, split_valid = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["label_transfer"],
    )
    logger.info(
        f"Transfer encoder train/valid split: train={len(split_train):,}, valid={len(split_valid):,}, "
        + f"positive_rate={split_train['label_transfer'].mean():.3f}"
    )

    resolved_device = _resolve_device(cfg.transfer_device)
    device = torch.device(resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.transfer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.transfer_model_name, num_labels=1)
    model.to(device)

    if resolved_device == "cpu":
        base_model = getattr(model, model.base_model_prefix, None)
        if base_model is not None:
            for p in base_model.parameters():
                p.requires_grad = False
        logger.info("CPU detected: froze transformer encoder and train classification head only")

    train_enc = tokenizer(
        split_train["source_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=int(cfg.transfer_max_len),
        return_tensors="pt",
    )
    valid_enc = tokenizer(
        split_valid["source_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=int(cfg.transfer_max_len),
        return_tensors="pt",
    )

    y_train = torch.tensor(split_train["label_transfer"].to_numpy(dtype=np.float32))
    y_valid = split_valid["label_transfer"].to_numpy(dtype=np.int64)
    train_ds = _build_dataset(train_enc, y_train, torch)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.transfer_batch_size), shuffle=True)

    pos = max(int(split_train["label_transfer"].sum()), 1)
    neg = max(int(len(split_train) - pos), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.transfer_lr),
    )

    model.train()
    for epoch in range(int(cfg.transfer_epochs)):
        losses: list[float] = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            logits = out.logits.view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        logger.info(f"Transfer epoch {epoch + 1}/{cfg.transfer_epochs} loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        valid_batch = {k: v.to(device) for k, v in valid_enc.items()}
        logits = model(**valid_batch).logits.view(-1).detach().cpu().numpy().astype("float64")
    probs = 1.0 / (1.0 + np.exp(-logits))
    threshold, _ = _select_threshold(y_valid, probs)
    metrics = _compute_metrics(y_valid, probs, threshold=threshold)
    metrics["n_train"] = int(len(split_train))
    metrics["n_valid"] = int(len(split_valid))
    metrics["model_name"] = cfg.transfer_model_name
    metrics["device"] = resolved_device

    logger.info(
        "Transfer source metrics: "
        + ", ".join(
            [
                f"AUC={metrics['auc']:.4f}" if not np.isnan(metrics["auc"]) else "AUC=nan",
                f"F1={metrics['f1']:.4f}",
                f"PR-AUC={metrics['pr_auc']:.4f}" if not np.isnan(metrics["pr_auc"]) else "PR-AUC=nan",
                f"ACC={metrics['accuracy']:.4f}",
                f"thr={metrics['threshold']:.2f}",
            ]
        )
    )

    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": cfg.transfer_model_name,
                "threshold": float(threshold),
                "max_len": int(cfg.transfer_max_len),
                "device": resolved_device,
                "epochs": int(cfg.transfer_epochs),
                "batch_size": int(cfg.transfer_batch_size),
                "lr": float(cfg.transfer_lr),
                "seed": int(seed),
            },
            f,
            indent=2,
        )
    write_csv(pd.DataFrame([metrics]), metrics_path)

    return TransferEncoderBundle(
        model_dir=model_dir,
        model_name=cfg.transfer_model_name,
        threshold=float(threshold),
        device=resolved_device,
        max_len=int(cfg.transfer_max_len),
        model=model,
        tokenizer=tokenizer,
    )
