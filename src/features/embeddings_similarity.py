from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingSimilarityResult:
    features: pd.DataFrame
    method: str


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    text = text or ""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else [""]

    words = text.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for w in words:
        extra = len(w) + (1 if current else 0)
        if current and (current_len + extra) > max_chars:
            chunks.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def compute_embedding_similarity_features(
    df: pd.DataFrame,
    *,
    text_col: str,
    seed_statements: tuple[str, ...],
    model_name: str,
    max_chars_per_chunk: int,
    model_dir: Path,
    logger,
) -> EmbeddingSimilarityResult:
    texts = df[text_col].fillna("").tolist()

    try:
        from joblib import dump
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        seed_emb = model.encode(list(seed_statements), batch_size=32, show_progress_bar=False, normalize_embeddings=True)

        doc_embs = []
        for t in texts:
            chunks = _chunk_text(t, max_chars=max_chars_per_chunk)
            emb = model.encode(chunks, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
            doc_embs.append(np.mean(emb, axis=0))
        doc_emb = np.vstack(doc_embs)

        sims = _cosine_sim(doc_emb, seed_emb)
        out = pd.DataFrame(
            {
                "ai_sim_mean": pd.Series(sims.mean(axis=1), dtype="float64"),
                "ai_sim_max": pd.Series(sims.max(axis=1), dtype="float64"),
            }
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        dump({"model_name": model_name, "seed_statements": seed_statements}, model_dir / "embeddings_meta.joblib")
        np.save(model_dir / "seed_embeddings.npy", seed_emb)

        logger.info("Embedding similarity computed with sentence-transformers")
        return EmbeddingSimilarityResult(features=out, method="sentence-transformers")
    except Exception as e:
        logger.info(f"sentence-transformers unavailable or failed ({e}); falling back to TF-IDF similarity")

    from joblib import dump
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_texts = list(seed_statements) + texts
    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.95, ngram_range=(1, 2))
    X = vec.fit_transform(all_texts)
    seed_X = X[: len(seed_statements)]
    doc_X = X[len(seed_statements) :]

    sims = (doc_X @ seed_X.T).toarray()
    out = pd.DataFrame(
        {
            "ai_sim_mean": pd.Series(sims.mean(axis=1), dtype="float64"),
            "ai_sim_max": pd.Series(sims.max(axis=1), dtype="float64"),
        }
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    dump(vec, model_dir / "tfidf_seed_vectorizer.joblib")
    logger.info("Embedding similarity computed with TF-IDF fallback")
    return EmbeddingSimilarityResult(features=out, method="tfidf-fallback")

