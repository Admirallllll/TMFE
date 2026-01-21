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


def _quarter_axis(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    q = (
        df[["quarter_index", "datacqtr"]]
        .dropna()
        .drop_duplicates()
        .sort_values("quarter_index", kind="mergesort")
    )
    x = q["quarter_index"].astype(int).to_numpy()
    labels = q["datacqtr"].astype(str).tolist()
    return x, labels


def _set_quarter_ticks(ax, x: np.ndarray, labels: list[str], *, max_ticks: int = 24) -> None:
    if len(x) == 0:
        return
    step = max(1, int(np.ceil(len(x) / max_ticks)))
    idx = np.arange(0, len(x), step)
    ax.set_xticks(x[idx])
    ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=7)


def plot_counts_by_quarter(df: pd.DataFrame, out_path: Path) -> None:
    counts = df.groupby(["quarter_index", "datacqtr"], sort=True).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(10, 4))
    x, labels = _quarter_axis(counts)
    ax.plot(counts["quarter_index"].astype(int), counts["n"], color="black", linewidth=1.5)
    ax.set_title("Transcript counts by quarter")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Count")
    _set_quarter_ticks(ax, x, labels)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_length_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    tokens = pd.to_numeric(df["n_tokens"], errors="coerce").fillna(0).astype(float).to_numpy()
    cap = float(np.nanquantile(tokens, 0.99)) if len(tokens) else 0.0
    x = np.clip(tokens, 0.0, cap)
    ax.hist(x, bins=50, color="gray", edgecolor="white")
    ax.set_title("Transcript length distribution (tokens, clipped at 99th pct)")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_ai_trend_over_time(df: pd.DataFrame, out_path: Path, *, by_sector: bool) -> None:
    x, labels = _quarter_axis(df)

    if by_sector:
        grp = (
            df.groupby(["sector", "quarter_index", "datacqtr"], sort=True)["ani_kw_per1k"]
            .mean()
            .reset_index()
        )
        top_sectors = (
            df["sector"].value_counts().head(6).index.tolist()
            if "sector" in df.columns
            else []
        )
        grp = grp.loc[grp["sector"].isin(top_sectors)]
        fig, ax = plt.subplots(figsize=(10, 5))
        for sector, g in grp.groupby("sector"):
            s = g.set_index(g["quarter_index"].astype(int))["ani_kw_per1k"].reindex(x).astype(float)
            ax.plot(x, s.to_numpy(), label=str(sector), linewidth=1.2)
        ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
        ax.set_title("AI keyword intensity trend by sector (top 6 by volume)")
    else:
        grp = df.groupby(["quarter_index", "datacqtr"], sort=True)["ani_kw_per1k"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(grp["quarter_index"].astype(int), grp["ani_kw_per1k"], color="tab:blue", linewidth=1.8)
        ax.set_title("AI keyword intensity trend (overall)")

    ax.set_xlabel("Quarter")
    ax.set_ylabel("ANI keywords per 1k tokens")
    _set_quarter_ticks(ax, x, labels)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_sector_quarter_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    pivot = (
        df.pivot_table(index="sector", columns="datacqtr", values="ani_kw_per1k", aggfunc="mean")
        .fillna(0.0)
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    mat = pivot.to_numpy()
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title("Sector x Quarter heatmap of ANI (mean per 1k tokens)")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=90, fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("ANI per 1k tokens")
    fig.tight_layout()
    _save(fig, out_path)


def plot_topic_trend(df: pd.DataFrame, out_path: Path, *, focus_from_datacqtr: str | None = "2022Q4") -> None:
    g = df.groupby(["quarter_index", "datacqtr"], sort=True)["ai_topic_share"].mean().reset_index()
    if focus_from_datacqtr is not None:
        g = g.loc[g["datacqtr"].astype(str) >= focus_from_datacqtr]
    x, labels = _quarter_axis(g)
    fig, ax = plt.subplots(figsize=(10, 4))
    y = g["ai_topic_share"].astype(float).to_numpy()
    ax.plot(g["quarter_index"].astype(int), y, color="tab:green", linewidth=1.8)
    ax.set_title("AI topic share trend")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("AI topic share")
    _set_quarter_ticks(ax, x, labels)
    if len(y) > 0 and float(np.nanmax(y) - np.nanmin(y)) < 1e-12:
        ax.text(
            0.5,
            0.5,
            "No variation in ai_topic_share\n(check topic identification or sample size)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="tab:red",
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_ani_vs_uncertainty(df: pd.DataFrame, out_path: Path) -> None:
    x = df["ani_kw_per1k"].astype(float)
    y = df["lm_unc_per1k"].astype(float)
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(x, y, gridsize=40, cmap="magma", mincnt=1)
    ax.set_title("ANI vs LM uncertainty (hexbin)")
    ax.set_xlabel("ANI keywords per 1k tokens")
    ax.set_ylabel("LM uncertainty per 1k tokens")
    cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Count")
    fig.tight_layout()
    _save(fig, out_path)


def generate_eda_figures(df: pd.DataFrame, *, figures_dir: Path, logger) -> list[Path]:
    paths = [
        figures_dir / "eda_counts_by_quarter.png",
        figures_dir / "eda_length_distribution_tokens.png",
        figures_dir / "eda_ai_trend_overall.png",
        figures_dir / "eda_ai_trend_by_sector.png",
        figures_dir / "eda_heatmap_sector_quarter_ani.png",
        figures_dir / "eda_ai_topic_share_trend.png",
        figures_dir / "eda_ani_vs_uncertainty_hexbin.png",
    ]

    plot_counts_by_quarter(df, paths[0])
    plot_length_distribution(df, paths[1])
    plot_ai_trend_over_time(df, paths[2], by_sector=False)
    plot_ai_trend_over_time(df, paths[3], by_sector=True)

    if "sector" in df.columns and "datacqtr" in df.columns:
        plot_sector_quarter_heatmap(df, paths[4])

    if "ai_topic_share" in df.columns:
        plot_topic_trend(df, paths[5])

    if "lm_unc_per1k" in df.columns and "ani_kw_per1k" in df.columns:
        plot_ani_vs_uncertainty(df, paths[6])

    logger.info(f"Saved EDA figures: {len(paths)}")
    return paths
