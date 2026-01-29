"""Generate visualizations from model_results.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regression_comparison(results: pd.DataFrame, out_path: Path) -> None:
    """Bar chart comparing regression models by RMSE and MAE."""
    reg = results.loc[results["task"] == "regression"].copy()
    if reg.empty:
        return

    reg = reg.sort_values("rmse", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # RMSE comparison
    colors = ["tab:green" if "BM" not in m else "tab:gray" for m in reg["model"]]
    axes[0].barh(reg["model"], reg["rmse"], color=colors, alpha=0.85)
    axes[0].set_title("Regression: RMSE (lower is better)")
    axes[0].set_xlabel("RMSE")
    if len(reg) > 0:
        axes[0].axvline(reg["rmse"].iloc[0], color="red", linestyle="--", alpha=0.5)

    # MAE comparison
    reg_mae = reg.sort_values("mae", ascending=True)
    colors = ["tab:green" if "BM" not in m else "tab:gray" for m in reg_mae["model"]]
    axes[1].barh(reg_mae["model"], reg_mae["mae"], color=colors, alpha=0.85)
    axes[1].set_title("Regression: MAE (lower is better)")
    axes[1].set_xlabel("MAE")

    fig.tight_layout()
    _save(fig, out_path)


def plot_classification_comparison(results: pd.DataFrame, out_path: Path) -> None:
    """Bar chart comparing classification models by AUC and F1."""
    cls = results.loc[results["task"] == "classification"].copy()
    if cls.empty:
        return

    cls = cls.sort_values("auc", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # AUC comparison
    colors = ["tab:green" if "BM" not in m else "tab:gray" for m in cls["model"]]
    axes[0].barh(cls["model"], cls["auc"], color=colors, alpha=0.85)
    axes[0].set_title("Classification: AUC (higher is better)")
    axes[0].set_xlabel("AUC")
    axes[0].set_xlim(0.4, 0.8)

    # F1 comparison
    cls_f1 = cls.sort_values("f1", ascending=False)
    colors = ["tab:green" if "BM" not in m else "tab:gray" for m in cls_f1["model"]]
    axes[1].barh(cls_f1["model"], cls_f1["f1"], color=colors, alpha=0.85)
    axes[1].set_title("Classification: F1 (higher is better)")
    axes[1].set_xlabel("F1 Score")

    fig.tight_layout()
    _save(fig, out_path)


def plot_delta_vs_benchmark(results: pd.DataFrame, out_path: Path) -> None:
    """Show improvement of text models vs metadata-only benchmark."""
    reg = results.loc[results["task"] == "regression"].copy()
    cls = results.loc[results["task"] == "classification"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Regression: RMSE reduction vs BM2
    if not reg.empty and "BM2_metadata_ridge" in reg["model"].values:
        bm2_rmse = float(reg.loc[reg["model"] == "BM2_metadata_ridge", "rmse"].iloc[0])
        text_models = reg.loc[~reg["model"].str.contains("BM")].copy()
        if not text_models.empty:
            text_models["rmse_reduction"] = bm2_rmse - text_models["rmse"]
            text_models = text_models.sort_values("rmse_reduction", ascending=False)

            colors = ["tab:green" if d > 0 else "tab:red" for d in text_models["rmse_reduction"]]
            axes[0].barh(text_models["model"], text_models["rmse_reduction"], color=colors, alpha=0.85)
            axes[0].axvline(0, color="black", linewidth=1)
            axes[0].set_title("RMSE Reduction vs BM2 (positive = better)")
            axes[0].set_xlabel("ΔRMSE")
    else:
        axes[0].text(0.5, 0.5, "No regression data available", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("RMSE Reduction vs BM2")

    # Classification: AUC improvement vs BM2
    if not cls.empty and "BM2_metadata_logit" in cls["model"].values:
        bm2_auc = float(cls.loc[cls["model"] == "BM2_metadata_logit", "auc"].iloc[0])
        text_models = cls.loc[~cls["model"].str.contains("BM")].copy()
        if not text_models.empty:
            text_models["auc_improvement"] = text_models["auc"] - bm2_auc
            text_models = text_models.sort_values("auc_improvement", ascending=False)

            colors = ["tab:green" if d > 0 else "tab:red" for d in text_models["auc_improvement"]]
            axes[1].barh(text_models["model"], text_models["auc_improvement"], color=colors, alpha=0.85)
            axes[1].axvline(0, color="black", linewidth=1)
            axes[1].set_title("AUC Improvement vs BM2 (positive = better)")
            axes[1].set_xlabel("ΔAUC")
    else:
        axes[1].text(0.5, 0.5, "No classification data available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("AUC Improvement vs BM2")

    fig.tight_layout()
    _save(fig, out_path)


def plot_model_summary(results: pd.DataFrame, out_path: Path) -> None:
    """Summary table visualization of all model results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Prepare table data
    reg = results.loc[results["task"] == "regression", ["model", "rmse", "mae", "r2"]].copy()
    cls = results.loc[results["task"] == "classification", ["model", "auc", "f1", "accuracy"]].copy()

    if not reg.empty:
        reg = reg.round(4)
        table_data = reg.values.tolist()
        col_labels = ["Model", "RMSE", "MAE", "R²"]
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="upper center",
            cellLoc="center",
            colColours=["lightblue"] * 4,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax.set_title("Model Performance Summary", fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    _save(fig, out_path)


def generate_results_figures(results_path: Path, figures_dir: Path, *, logger=None) -> list[Path]:
    """Generate all figures from model_results.csv."""
    if not results_path.exists():
        if logger:
            logger.warning(f"Model results file not found: {results_path}")
        return []

    results = pd.read_csv(results_path)

    paths = [
        figures_dir / "results_model_compare_regression.png",
        figures_dir / "results_model_compare_classification.png",
        figures_dir / "results_delta_vs_benchmark.png",
    ]

    plot_regression_comparison(results, paths[0])
    plot_classification_comparison(results, paths[1])
    plot_delta_vs_benchmark(results, paths[2])

    created = [p for p in paths if p.exists()]
    if logger:
        logger.info(f"Generated {len(created)} model results figures")

    return created
