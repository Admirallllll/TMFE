from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path = project_root()
    data_dir: Path = root / "data"
    processed_dir: Path = data_dir / "processed"
    external_dir: Path = data_dir / "external"
    hf_cache_dir: Path = data_dir / "hf_cache"

    outputs_dir: Path = root / "outputs"
    figures_dir: Path = outputs_dir / "figures"
    tables_dir: Path = outputs_dir / "tables"
    models_dir: Path = outputs_dir / "models"
    logs_dir: Path = outputs_dir / "logs"


@dataclass(frozen=True)
class DatasetConfig:
    hf_name: str = "glopardo/sp500-earnings-transcripts"
    hf_split: str = "train"


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    dev_mode: bool = False
    dev_sample_n: int = 800
    recompute: bool = False

    train_end_datacqtr: str = "2022Q3"
    test_start_datacqtr: str = "2022Q4"

    remove_boilerplate: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    ai_terms_per1k_denom: str = "tokens"

    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.95
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_max_features: int | None = 50_000

    topic_model: str = "auto"
    lda_num_topics: int = 25
    lda_passes: int = 5
    lda_chunksize: int = 2000

    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_max_chars_per_chunk: int = 4000
    embeddings_max_chunks_per_doc: int | None = 12
    embeddings_batch_size: int = 64
    embeddings_device: str = "cuda"

    bertopic_calculate_probabilities: bool = True


AI_SEED_STATEMENTS: tuple[str, ...] = (
    "We are deploying generative AI to improve productivity and reduce costs.",
    "AI is driving new product capabilities and revenue growth.",
    "We are integrating large language models into our software and workflows.",
    "We are investing in machine learning to improve recommendations and personalization.",
    "We are focused on AI safety, ethics, and governance.",
    "We see risks from AI regulation, compliance, and data privacy.",
    "AI is helping automate customer support and internal operations.",
    "We are using AI to optimize supply chains and forecasting.",
)
