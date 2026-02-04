from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "company",
    "cik",
    "sector",
    "industry",
    "earnings_date",
    "datacqtr",
    "datafqtr",
    "year",
    "quarter",
    "eps12mtrailing_qavg",
    "eps12mtrailing_eoq",
    "eps12mfwd_qavg",
    "eps12mfwd_eoq",
    "eps_lt",
    "peforw_qavg",
    "peforw_eoq",
    "transcript",
)


def _missingness_summary(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append({"column": c, "missing_frac": 1.0})
            continue
        rows.append({"column": c, "missing_frac": float(df[c].isna().mean())})
    return pd.DataFrame(rows).sort_values("missing_frac", ascending=False)


def _sample_dev(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df
    if "ticker" not in df.columns:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    counts = df["ticker"].value_counts().sort_index()
    tickers = counts.index.to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(tickers)

    chosen: list[str] = []
    total = 0
    for t in tickers:
        chosen.append(t)
        total += int(counts.loc[t])
        if total >= n:
            break

    sampled = df.loc[df["ticker"].isin(chosen)].reset_index(drop=True)
    return sampled


def load_hf_dataset_to_df(
    dataset_name: str,
    split: str,
    cache_dir: Path,
    *,
    dev_mode: bool,
    dev_sample_n: int,
    seed: int,
    logger,
    start_year: int | None = None,
) -> pd.DataFrame:
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading Hugging Face dataset: {dataset_name} [{split}]")
    download_config = None
    try:
        from datasets import DownloadConfig

        download_config = DownloadConfig(max_retries=10)
    except Exception:
        download_config = None

    local_dir = os.environ.get("SP500_TRANSCRIPTS_LOCAL_DIR")
    local_df = None
    if local_dir:
        try:
            local_path = Path(local_dir).expanduser().resolve()
            parquet_files = sorted(local_path.glob("**/*.parquet"))
            if parquet_files:
                logger.info(f"Loading local parquet shards from SP500_TRANSCRIPTS_LOCAL_DIR={local_path}")
                ds_local = load_dataset(
                    "parquet",
                    data_files={"train": [str(p) for p in parquet_files]},
                    split="train",
                )
                local_df = ds_local.to_pandas()
        except Exception as e:
            logger.info(f"Local parquet load failed ({e}); ignoring local fallback")

    df = None
    if local_df is not None:
        df = local_df
    else:
        try:
            if dev_mode:
                import itertools

                try:
                    ds_stream = load_dataset(
                        dataset_name,
                        split=split,
                        cache_dir=str(cache_dir),
                        download_config=download_config,
                        streaming=True,
                    )
                    try:
                        ds_stream = ds_stream.shuffle(seed=seed, buffer_size=10_000)
                    except Exception:
                        pass
                    records = list(itertools.islice(ds_stream, dev_sample_n))
                    df = pd.DataFrame.from_records(records)
                    logger.info(f"Loaded {len(df):,} streamed rows for DEV_MODE")
                except Exception as e:
                    logger.info(f"Streaming DEV_MODE load failed ({e}); attempting full download")
                    ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir), download_config=download_config)
                    df = ds.to_pandas()
            else:
                ds = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir), download_config=download_config)
                df = ds.to_pandas()
        except Exception as e:
            msg = (
                "Failed to download dataset from Hugging Face Hub. "
                "This dataset may require network access to Hugging Face's large-file backend. "
                "If your network blocks these hosts, download the parquet shards manually and set "
                "SP500_TRANSCRIPTS_LOCAL_DIR to the folder containing them. "
                f"Original error: {e}"
            )
            raise RuntimeError(msg) from e

    logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    missing = _missingness_summary(df, REQUIRED_COLUMNS)
    logger.info("Missingness (required cols, top 8):\n" + missing.head(8).to_string(index=False))

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Filter by start_year if specified
    if start_year is not None:
        before_filter = len(df)
        df["_year"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.year
        df = df.loc[df["_year"] >= start_year].drop(columns=["_year"]).reset_index(drop=True)
        logger.info(f"Filtered to start_year >= {start_year}: {before_filter:,} -> {len(df):,} rows")

    if dev_mode and len(df) > dev_sample_n:
        df = _sample_dev(df, dev_sample_n, seed)
        logger.info(f"DEV_MODE sampling applied: {len(df):,} rows")

    return df
