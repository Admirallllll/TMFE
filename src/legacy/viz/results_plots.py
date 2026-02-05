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


def plot_model_results(results: pd.DataFrame, out_path: Path) -> None:
    reg = results.loc[results["task"] == "regression"].copy()
    cls = results.loc[results["task"] == "classification"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if not reg.empty:
        reg = reg.sort_values("rmse", ascending=True)
        axes[0].barh(reg["model"], reg["rmse"], color="tab:blue", alpha=0.8)
        axes[0].set_title("Regression (lower RMSE is better)")
        axes[0].set_xlabel("RMSE")
    else:
        axes[0].axis("off")

    if not cls.empty:
        cls = cls.sort_values("auc", ascending=False)
        axes[1].barh(cls["model"], cls["auc"], color="tab:green", alpha=0.8)
        axes[1].set_title("Classification (higher AUC is better)")
        axes[1].set_xlabel("AUC")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    _save(fig, out_path)


def plot_pre_post_sector(pre_post: pd.DataFrame, out_path: Path, *, metric_col: str) -> None:
    d = pre_post.copy()
    d = d.loc[d["group"] == "sector"]
    d = d.sort_values(["metric", "sector", "period"])
    d = d.loc[d["metric"] == metric_col]

    sectors = d["sector"].unique().tolist()
    pre = d.loc[d["period"] == "pre"].set_index("sector")["value"]
    post = d.loc[d["period"] == "post"].set_index("sector")["value"]
    pre = pre.reindex(sectors).fillna(0.0)
    post = post.reindex(sectors).fillna(0.0)

    idx = np.arange(len(sectors))
    width = 0.42
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(idx - width / 2, pre.values, width=width, label="pre", color="gray")
    ax.bar(idx + width / 2, post.values, width=width, label="post", color="tab:orange")
    ax.set_xticks(idx)
    ax.set_xticklabels(sectors, rotation=90, fontsize=7)
    ax.set_ylabel(metric_col)
    ax.set_title(f"Pre vs Post {metric_col} by sector")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    vals = np.concatenate([pre.values.astype(float), post.values.astype(float)]) if len(sectors) else np.array([])
    if len(vals) > 0 and float(np.nanmax(vals) - np.nanmin(vals)) < 1e-12:
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5,
            0.5,
            f"No variation in {metric_col} (all values equal).\nRecompute features or increase sample size.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="tab:red",
        )

    fig.tight_layout()
    _save(fig, out_path)


def plot_sector_heterogeneity(sector_table: pd.DataFrame, out_path: Path, *, metric_col: str) -> None:
    d = sector_table.copy()
    d = d.loc[d["metric"] == metric_col].sort_values("delta", ascending=False)
    d = d.head(12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(d["sector"], d["delta"], color="tab:purple", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title(f"Sector heterogeneity: Δ{metric_col} (M1 - BM2)")
    ax.set_xlabel(f"Δ{metric_col}")
    fig.tight_layout()
    _save(fig, out_path)
