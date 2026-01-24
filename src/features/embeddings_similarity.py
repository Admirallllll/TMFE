from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.document_embeddings import compute_document_embeddings, resolve_device


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
    max_chunks_per_doc: int | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    model_dir: Path,
    logger,
    doc_embeddings: np.ndarray | None = None,
) -> EmbeddingSimilarityResult:
    try:
        from joblib import dump
        from sentence_transformers import SentenceTransformer

        if doc_embeddings is not None:
            doc_emb = np.asarray(doc_embeddings)
            if doc_emb.ndim != 2 or doc_emb.shape[0] != len(df):
                raise ValueError("doc_embeddings shape mismatch")
        else:
            doc_emb = compute_document_embeddings(
                df,
                text_col=text_col,
                model_name=model_name,
                max_chars_per_chunk=max_chars_per_chunk,
                max_chunks_per_doc=max_chunks_per_doc,
                batch_size=batch_size,
                device=device,
                logger=logger,
            ).embeddings

        resolved_device = resolve_device(device, logger=logger)
        model = SentenceTransformer(model_name, device=resolved_device)
        seed_emb = model.encode(
            list(seed_statements),
            batch_size=min(int(batch_size), 64),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32", copy=False)

        sims = _cosine_sim(doc_emb.astype("float32", copy=False), seed_emb)
        out = pd.DataFrame(
            {
                "ai_sim_mean": pd.Series(sims.mean(axis=1), dtype="float64"),
                "ai_sim_max": pd.Series(sims.max(axis=1), dtype="float64"),
            }
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        dump({"model_name": model_name, "seed_statements": seed_statements, "device": resolved_device}, model_dir / "embeddings_meta.joblib")
        np.save(model_dir / "seed_embeddings.npy", seed_emb)

        logger.info("Embedding similarity computed with sentence-transformers")
        return EmbeddingSimilarityResult(features=out, method="sentence-transformers")
    except Exception as e:
        logger.info(f"sentence-transformers unavailable or failed ({e}); falling back to TF-IDF similarity")

    from joblib import dump
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = df[text_col].fillna("").astype(str).tolist()
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
