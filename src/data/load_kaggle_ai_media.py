from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Paths
from src.features.transfer_tag_labeling import parse_tags

DATASET_REF = "jannalipenkova/ai-media-dataset"
TABLE_SUFFIXES: tuple[str, ...] = (".csv", ".parquet", ".json", ".jsonl", ".ndjson")
TEXT_CANDIDATES: tuple[str, ...] = (
    "text",
    "content",
    "article",
    "body",
    "post",
    "description",
    "summary",
    "title",
)
TAG_CANDIDATES: tuple[str, ...] = ("tags", "tag", "labels", "topics")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table suffix: {path}")


def _pick_column(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    col_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in col_map:
            return col_map[c]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None


def _extract_tags(value: object) -> list[str]:
    return parse_tags(value)


def _extract_from_files(files: list[Path], *, logger, source_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = _read_table(path)
        except Exception as e:
            logger.info(f"Skipping unreadable file {path.name}: {e}")
            continue
        if df.empty:
            continue

        text_col = _pick_column(df.columns.tolist(), TEXT_CANDIDATES)
        tags_col = _pick_column(df.columns.tolist(), TAG_CANDIDATES)
        if text_col is None or tags_col is None:
            logger.info(f"Skipping file {path.name}: no text/tags columns detected")
            continue

        part = df[[text_col, tags_col]].copy()
        part.columns = ["source_text", "tags_raw"]
        part["source_text"] = part["source_text"].fillna("").astype(str).str.strip()
        part["tags_norm"] = part["tags_raw"].map(_extract_tags)
        part = part.loc[(part["source_text"] != "") & (part["tags_norm"].map(len) > 0)].copy()
        if part.empty:
            continue

        part["source_file"] = str(path)
        part["source_origin"] = source_name
        frames.append(part.reset_index(drop=True))
        logger.info(f"Accepted {source_name} file {path.name} with {len(part):,} usable rows")

    if not frames:
        return pd.DataFrame(columns=["source_text", "tags_raw", "tags_norm", "source_file", "source_origin"])

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["source_text", "tags_raw"]).reset_index(drop=True)
    return out


def load_ai_media_dataset(
    logger,
    *,
    local_candidates: tuple[Path, ...] | None = None,
) -> pd.DataFrame:
    if local_candidates is None:
        local_candidates = (
            Paths().data_dir / "ai_media_dataset_20250911.csv",
            Paths().data_dir / "ai_media_dataset.csv",
        )

    local_files = [Path(p).expanduser().resolve() for p in local_candidates if Path(p).expanduser().exists()]
    if local_files:
        logger.info(f"Local AI media dataset detected; using local files: {', '.join(str(p) for p in local_files)}")
        local_df = _extract_from_files(local_files, logger=logger, source_name="local")
        if local_df.empty:
            raise RuntimeError(
                "Local AI media file exists but could not parse usable text+tags rows. "
                f"Checked files: {', '.join(str(p) for p in local_files)}"
            )
        logger.info(f"AI media dataset prepared from local data: {len(local_df):,} rows")
        return local_df

    try:
        import kagglehub
    except Exception as e:
        raise RuntimeError(
            "Local AI media dataset not found, and Kaggle download is unavailable. "
            "Please configure Kaggle credentials (kaggle.json) and install kagglehub, "
            f"or place a local file at {Paths().data_dir / 'ai_media_dataset_20250911.csv'}. "
            "Kaggle credentials are required for auto-download."
        ) from e

    try:
        dataset_dir = Path(kagglehub.dataset_download(DATASET_REF)).resolve()
    except Exception as e:
        raise RuntimeError(
            "Failed to download AI media dataset from Kaggle. "
            "Please verify Kaggle credentials are configured correctly and retry. "
            "Kaggle credentials are required when no local dataset file is available."
        ) from e
    logger.info(f"Downloaded/located AI media dataset at: {dataset_dir}")

    files = sorted([p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES])
    if not files:
        raise RuntimeError(f"No readable table files found in dataset directory: {dataset_dir}")

    out = _extract_from_files(files, logger=logger, source_name="kaggle")
    if out.empty:
        raise RuntimeError("No file with both text and tags columns could be parsed from AI Media dataset.")
    logger.info(f"AI media dataset prepared: {len(out):,} rows")
    return out
