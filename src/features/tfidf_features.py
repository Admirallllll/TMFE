from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TfidfArtifacts:
    vectorizer_path: Path
    matrix_path: Path
    vocab_size: int


def fit_tfidf(
    df: pd.DataFrame,
    *,
    text_col: str,
    model_dir: Path,
    min_df: int,
    max_df: float,
    ngram_range: tuple[int, int],
    max_features: int | None,
    logger,
) -> TfidfArtifacts:
    from joblib import dump
    from scipy.sparse import save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = df[text_col].fillna("").tolist()
    vec = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    X = vec.fit_transform(texts)

    model_dir.mkdir(parents=True, exist_ok=True)
    vec_path = model_dir / "tfidf_vectorizer.joblib"
    mat_path = model_dir / "tfidf_matrix.npz"
    dump(vec, vec_path)
    save_npz(mat_path, X)

    logger.info(f"TF-IDF fitted: vocab_size={len(vec.vocabulary_):,}, matrix_shape={X.shape}")
    return TfidfArtifacts(vectorizer_path=vec_path, matrix_path=mat_path, vocab_size=len(vec.vocabulary_))


def top_ngrams(vectorizer_path: Path, matrix_path: Path, *, top_k: int = 30) -> pd.DataFrame:
    from joblib import load

    vec = load(vectorizer_path)
    if matrix_path.suffix.lower() == ".npz":
        from scipy.sparse import load_npz

        X = load_npz(matrix_path)
    else:
        X = load(matrix_path)
    means = np.asarray(X.mean(axis=0)).ravel()
    idx = np.argsort(-means)[:top_k]
    inv_vocab = {i: t for t, i in vec.vocabulary_.items()}
    return pd.DataFrame({"ngram": [inv_vocab[i] for i in idx], "mean_tfidf": means[idx]})
