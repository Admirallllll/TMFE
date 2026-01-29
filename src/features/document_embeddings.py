from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DocumentEmbeddingsResult:
    embeddings: np.ndarray
    method: str
    device: str
    n_chunks: int


def resolve_device(device: str, *, logger=None) -> str:
    preferred = (device or "").strip().lower()
    if preferred in {"", "auto"}:
        preferred = "cuda"

    try:
        import torch
    except Exception:
        if logger is not None:
            logger.info("PyTorch not available; falling back to CPU for embeddings")
        return "cpu"

    if preferred.startswith("cuda"):
        if torch.cuda.is_available():
            return device
        if logger is not None:
            logger.info("CUDA requested but not available; falling back to CPU for embeddings")
        return "cpu"

    if preferred == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if logger is not None:
            logger.info("MPS requested but not available; falling back to CPU for embeddings")
        return "cpu"

    return device


def _chunk_text_by_words(text: str, *, max_chars: int, max_chunks: int | None) -> list[str]:
    """Chunk text at word boundaries to avoid splitting words mid-token.

    This preserves semantic integrity of the text for embedding computation.
    """
    text = (text or "").strip()
    if not text:
        return [""]
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for w in words:
        word_len = len(w)
        # +1 for space between words
        if current_len + word_len + (1 if current_words else 0) > max_chars and current_words:
            chunks.append(" ".join(current_words))
            current_words = [w]
            current_len = word_len
        else:
            current_words.append(w)
            current_len += word_len + (1 if len(current_words) > 1 else 0)

    if current_words:
        chunks.append(" ".join(current_words))

    if not chunks:
        return [""]

    if max_chunks is None or len(chunks) <= max_chunks:
        return chunks

    # Sample chunks evenly if we have too many
    idx = np.linspace(0, len(chunks) - 1, num=int(max_chunks), dtype=int)
    return [chunks[i] for i in idx]


def compute_document_embeddings(
    df: pd.DataFrame,
    *,
    text_col: str,
    model_name: str,
    max_chars_per_chunk: int,
    batch_size: int,
    device: str,
    max_chunks_per_doc: int | None,
    logger,
) -> DocumentEmbeddingsResult:
    from sentence_transformers import SentenceTransformer

    resolved_device = resolve_device(device, logger=logger)

    texts = df[text_col].fillna("").astype(str).tolist()
    n_docs = len(texts)
    chunk_batch_size = max(1024, int(batch_size) * 64)

    if logger is not None:
        logger.info(
            "Computing document embeddings: "
            + ", ".join(
                [
                    f"docs={n_docs:,}",
                    f"model={model_name}",
                    f"device={resolved_device}",
                    f"batch_size={batch_size}",
                    f"chunk_batch_size={chunk_batch_size}",
                    f"max_chunks_per_doc={max_chunks_per_doc if max_chunks_per_doc is not None else 'none'}",
                ]
            )
        )

    model = SentenceTransformer(model_name, device=resolved_device)

    doc_sum: np.ndarray | None = None
    counts = np.zeros((n_docs,), dtype="int32")

    batch_chunks: list[str] = []
    batch_doc_ids: list[int] = []
    n_chunks = 0

    def _flush_batch() -> None:
        nonlocal doc_sum, batch_chunks, batch_doc_ids
        if not batch_chunks:
            return
        emb = model.encode(
            batch_chunks,
            batch_size=int(batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32", copy=False)
        if doc_sum is None:
            doc_sum = np.zeros((n_docs, int(emb.shape[1])), dtype="float32")
        doc_ids = np.asarray(batch_doc_ids, dtype="int32")
        np.add.at(doc_sum, doc_ids, emb)
        np.add.at(counts, doc_ids, 1)
        batch_chunks = []
        batch_doc_ids = []

    for i, t in enumerate(texts):
        chunks = _chunk_text_by_words(t, max_chars=max_chars_per_chunk, max_chunks=max_chunks_per_doc)
        for c in chunks:
            batch_chunks.append(c)
            batch_doc_ids.append(i)
            n_chunks += 1
            if len(batch_chunks) >= chunk_batch_size:
                _flush_batch()
        if logger is not None and (i + 1) % 2000 == 0:
            logger.info(f"Embeddings progress: {i+1:,}/{n_docs:,} docs")

    _flush_batch()

    if doc_sum is None:
        doc_sum = np.zeros((n_docs, 1), dtype="float32")
    doc_emb = doc_sum / np.maximum(counts[:, None], 1)

    return DocumentEmbeddingsResult(
        embeddings=doc_emb,
        method="sentence-transformers",
        device=resolved_device,
        n_chunks=int(n_chunks),
    )
