from __future__ import annotations

from pathlib import Path

import pandas as pd


def _summarize_df(df: pd.DataFrame) -> str:
    null_rates = df.isna().mean().sort_values(ascending=False)
    top_nulls = null_rates.head(5).to_dict()
    return f"shape={df.shape}, null_rates_top5={top_nulls}, head=\n{df.head(3).to_string(index=False)}"


def _require_nonempty(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        raise ValueError(f"{name} is empty. {_summarize_df(df)}")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_trend_line(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    logger=None,
) -> None:
    _require_nonempty(df, f"plot_trend_line:{out_path.name}")
    if df[y_col].isna().all():
        raise ValueError(f"{out_path.name} {y_col} is all NaN. {_summarize_df(df)}")
    if logger is not None:
        logger.info(f"Plot input {out_path.name}: {_summarize_df(df)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[x_col].astype(str), df[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stacked_bar(
    df: pd.DataFrame,
    *,
    index_col: str,
    category_col: str,
    value_col: str,
    title: str,
    out_path: Path,
    logger=None,
) -> None:
    _require_nonempty(df, f"plot_stacked_bar:{out_path.name}")
    if df[value_col].isna().all():
        raise ValueError(f"{out_path.name} {value_col} is all NaN. {_summarize_df(df)}")
    if logger is not None:
        logger.info(f"Plot input {out_path.name}: {_summarize_df(df)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index=index_col, columns=category_col, values=value_col, aggfunc="sum", fill_value=0)
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(index_col)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bar(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    logger=None,
) -> None:
    _require_nonempty(df, f"plot_bar:{out_path.name}")
    if df[y_col].isna().all():
        raise ValueError(f"{out_path.name} {y_col} is all NaN. {_summarize_df(df)}")
    if logger is not None:
        logger.info(f"Plot input {out_path.name}: {_summarize_df(df)}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_placeholder(out_path: Path, *, reason: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, f"skipped: {reason}", ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
