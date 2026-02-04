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
    transfer_data_dir: Path = external_dir / "transfer"
    hf_cache_dir: Path = data_dir / "hf_cache"

    outputs_dir: Path = root / "outputs"
    figures_dir: Path = outputs_dir / "figures"
    tables_dir: Path = outputs_dir / "tables"
    models_dir: Path = outputs_dir / "models"
    logs_dir: Path = outputs_dir / "logs"
    transfer_model_dir: Path = models_dir / "transfer"
    transfer_table_dir: Path = tables_dir / "transfer"


@dataclass(frozen=True)
class DatasetConfig:
    hf_name: str = "glopardo/sp500-earnings-transcripts"
    hf_split: str = "train"
    start_year: int | None = 2018  # Filter data to start from this year (None = no filter)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    dev_mode: bool = False
    dev_sample_n: int = 800
    recompute: bool = False

    train_end_datacqtr: str = "2022Q3"
    test_start_datacqtr: str = "2022Q4"

    remove_boilerplate: bool = False
    transfer_retrain: bool = False
    transfer_max_train_samples: int | None = None


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

    transfer_enabled: bool = True
    transfer_model_name: str = "distilroberta-base"
    transfer_epochs: int = 2
    transfer_lr: float = 2e-5
    transfer_batch_size: int = 16
    transfer_max_len: int = 256
    transfer_device: str = "auto"
    transfer_min_tag_freq: int = 20
    transfer_label_margin: int = 3


@dataclass(frozen=True)
class FeatureColumns:
    """Centralized registry of feature column names used in modeling.

    This prevents hardcoding column names across multiple files and makes
    it easier to add/remove features consistently.
    """

    # Metadata features (lagged financials + document stats)
    meta_numeric: tuple[str, ...] = (
        "n_tokens",
        "n_chars",
        "lag_peforw_qavg",
        "lag_eps12mtrailing_qavg",
        "lag_eps12mtrailing_eoq",
        "lag_eps12mfwd_qavg",
        "lag_eps12mfwd_eoq",
        "lag_eps_lt",
    )

    # Text-derived features (ANI + sentiment + topics + embeddings)
    text_numeric: tuple[str, ...] = (
        "ani_kw_per1k",
        "ani_ai_core_per1k",
        "ani_ml_per1k",
        "ani_llm_per1k",
        "ani_genai_per1k",
        "lm_pos_per1k",
        "lm_neg_per1k",
        "lm_unc_per1k",
        "lm_net_tone_per1k",
        "ai_topic_share",
        "ai_sim_mean",
        "ai_sim_max",
        "transfer_ai_prob",
        "transfer_ai_logit",
        "transfer_ai_confidence",
    )

    # Categorical features
    categorical: tuple[str, ...] = ("sector", "datacqtr", "ticker")

    # Target columns
    target_regression: str = "delta_peforw_qavg"
    target_regression_robust: str = "delta_peforw_eoq"
    target_classification: str = "valuation_upgrade"

    @property
    def all_numeric_text(self) -> tuple[str, ...]:
        """Meta + text features combined for full text models."""
        return self.meta_numeric + self.text_numeric


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
