from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.ai_dictionary import ai_term_patterns


_AI_REGEXES = [re.compile(pat) for pat in ai_term_patterns().values()]


def compute_is_ai_kw(text_series: pd.Series) -> pd.Series:
    def _has_kw(text: str) -> bool:
        for rgx in _AI_REGEXES:
            if rgx.search(text):
                return True
        return False

    return text_series.fillna("").astype(str).apply(_has_kw)


def apply_ai_rules(row: dict[str, object], *, thr_hi: float, thr_lo: float) -> dict[str, object]:
    is_ai_kw = bool(row.get("is_ai_kw", False))
    score = float(row.get("ai_score_encoder", 0.0))
    needs_review = (not is_ai_kw) and (thr_lo <= score < thr_hi)
    is_ai_final = is_ai_kw or (score >= thr_hi)
    return {
        "is_ai_final": is_ai_final,
        "needs_review": needs_review,
    }


@dataclass(frozen=True)
class CalibrationResult:
    thr_hi: float
    thr_lo: float
    precision: float
    recall: float
    f1: float


def sample_for_calibration(turns_df: pd.DataFrame, *, n: int, seed: int, out_path: Path) -> Path:
    sample = turns_df.sample(n=min(n, len(turns_df)), random_state=seed).copy()
    sample["manual_label"] = ""
    sample.to_csv(out_path, index=False)
    return out_path


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def select_thresholds(
    labeled_csv: Path,
    *,
    score_col: str = "ai_score_encoder",
    label_col: str = "manual_label",
) -> CalibrationResult:
    df = pd.read_csv(labeled_csv)
    y_true = df[label_col].astype(int).to_numpy()
    scores = df[score_col].astype(float).to_numpy()

    best = CalibrationResult(0.8, 0.6, 0.0, 0.0, 0.0)
    thr_hi_candidates = np.arange(0.0, 1.01, 0.01)
    for thr_hi in thr_hi_candidates:
        y_pred = (scores >= thr_hi).astype(int)
        precision, recall, f1 = _compute_metrics(y_true, y_pred)
        if precision >= 0.85:
            if (f1 > best.f1) or (f1 == best.f1 and recall > best.recall):
                best = CalibrationResult(float(thr_hi), best.thr_lo, precision, recall, f1)

    if best.precision < 0.85:
        for thr_hi in thr_hi_candidates:
            y_pred = (scores >= thr_hi).astype(int)
            precision, recall, f1 = _compute_metrics(y_true, y_pred)
            if (precision > best.precision) or (precision == best.precision and f1 > best.f1):
                best = CalibrationResult(float(thr_hi), best.thr_lo, precision, recall, f1)

    thr_lo = None
    for thr in thr_hi_candidates:
        y_pred = (scores >= thr).astype(int)
        precision, recall, f1 = _compute_metrics(y_true, y_pred)
        if recall >= 0.90:
            thr_lo = float(thr)
    if thr_lo is None:
        thr_lo = max(0.0, best.thr_hi - 0.15)
    thr_lo = min(thr_lo, best.thr_hi)

    return CalibrationResult(best.thr_hi, thr_lo, best.precision, best.recall, best.f1)


def write_calibration_metrics(out_path: Path, result: CalibrationResult) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["thr_hi", "thr_lo", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "thr_hi": result.thr_hi,
                "thr_lo": result.thr_lo,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
            }
        )
