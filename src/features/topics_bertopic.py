from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.text import tokenize_simple


@dataclass(frozen=True)
class TopicResult:
    features: pd.DataFrame
    ai_topic_ids: set[int]
    method: str


def _doc_ai_score_from_text(docs: list[str]) -> np.ndarray:
    ai_tokens = {
        "ai",
        "ml",
        "llm",
        "llms",
        "genai",
        "chatgpt",
        "copilot",
        "transformer",
        "rag",
        "prompt",
        "prompts",
    }
    scores: list[float] = []
    for d in docs:
        toks = [t.lower() for t in tokenize_simple(d)]
        n = max(len(toks), 1)
        cnt = sum(1 for t in toks if t in ai_tokens)
        scores.append((cnt / n) * 1000.0)
    return np.asarray(scores, dtype="float64")


def _select_ai_topics_by_enrichment(
    assigned_topics: list[int],
    ai_score: np.ndarray,
    *,
    min_ratio: float = 1.5,
    top_k: int = 3,
    min_topic_docs: int = 15,
    logger=None,
) -> set[int]:
    if len(assigned_topics) == 0:
        return set()
    if ai_score is None or len(ai_score) != len(assigned_topics):
        return set()

    d = pd.DataFrame({"topic_id": pd.Series(assigned_topics, dtype="int32"), "ai_score": ai_score})
    d = d.loc[d["topic_id"] != -1].copy()
    if d.empty:
        return set()

    global_mean = float(d["ai_score"].mean())
    eps = 1e-9
    stats = (
        d.groupby("topic_id", sort=True)["ai_score"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n_docs"})
        .loc[lambda x: x["n_docs"] >= int(min_topic_docs)]
        .copy()
    )
    if stats.empty:
        return set()

    stats["ratio"] = (stats["mean"] + eps) / (global_mean + eps)
    enriched = stats.loc[(stats["ratio"] >= float(min_ratio)) & (stats["mean"] > 0.0)].copy()
    if enriched.empty:
        return set()

    enriched = enriched.sort_values(["ratio", "mean", "n_docs"], ascending=False)
    chosen = set(int(t) for t in enriched.head(int(top_k)).index.tolist())
    if logger is not None:
        msg = ", ".join(f"{t}(ratio={enriched.loc[t,'ratio']:.2f})" for t in sorted(chosen))
        logger.info(f"AI topics selected by enrichment: {msg}")
    return chosen


def _ai_topics_from_keywords(topic_keywords: dict[int, list[str]]) -> set[int]:
    ai_phrases = {
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "generative ai",
        "large language model",
        "large language models",
        "foundation model",
        "foundation models",
        "retrieval augmented",
        "retrieval-augmented",
        "chatgpt",
    }
    ai_kw_loose = {
        "ai",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "deep",
        "neural",
        "network",
        "generative",
        "genai",
        "llm",
        "llms",
        "chatgpt",
        "copilot",
        "transformer",
        "prompt",
        "prompts",
        "rag",
        "retrieval",
        "augmented",
        "foundation",
    }
    strong_tokens = {
        "ai",
        "llm",
        "llms",
        "genai",
        "chatgpt",
        "copilot",
        "transformer",
        "rag",
        "prompt",
        "prompts",
    }
    ai_topics: set[int] = set()
    for tid, words in topic_keywords.items():
        if tid == -1:
            continue
        kw = [w.lower() for w in words]
        kw_set = set(kw)
        if any(w in ai_phrases for w in kw):
            ai_topics.add(tid)
            continue
        if kw_set & strong_tokens:
            ai_topics.add(tid)
            continue
        if ("artificial" in kw_set and "intelligence" in kw_set) or ("machine" in kw_set and "learning" in kw_set):
            ai_topics.add(tid)
            continue
        if sum(1 for w in kw_set if w in ai_kw_loose) >= 2:
            ai_topics.add(tid)
    return ai_topics


def _fit_transform_bertopic(
    docs: list[str],
    *,
    model_dir: Path,
    logger,
) -> tuple[list[int], np.ndarray | None, dict[int, list[str]]]:
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from joblib import dump
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    topic_model = BERTopic(
        language="english",
        calculate_probabilities=True,
        verbose=False,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=15,
        min_topic_size=15,
    )
    topics, probs = topic_model.fit_transform(docs)

    topic_keywords: dict[int, list[str]] = {}
    for tid in topic_model.get_topics().keys():
        if tid == -1:
            continue
        kws = topic_model.get_topic(tid) or []
        topic_keywords[int(tid)] = [w for w, _ in kws[:20]]

    model_dir.mkdir(parents=True, exist_ok=True)
    dump(topic_model, model_dir / "bertopic_model.joblib")
    pd.DataFrame(
        [{"topic_id": k, "keywords": ", ".join(v)} for k, v in sorted(topic_keywords.items())]
    ).to_csv(model_dir / "bertopic_topics.csv", index=False)

    logger.info(f"BERTopic fitted with {len(topic_keywords):,} topics")
    return list(map(int, topics)), probs, topic_keywords


def _fit_transform_lda(
    docs: list[str],
    *,
    num_topics: int,
    passes: int,
    chunksize: int,
    model_dir: Path,
    logger,
) -> tuple[list[int], list[list[tuple[int, float]]], dict[int, list[str]]]:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    from joblib import dump
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = set(ENGLISH_STOP_WORDS)
    keep_tokens = {
        "ai",
        "ml",
        "llm",
        "llms",
        "genai",
        "chatgpt",
        "copilot",
        "transformer",
        "prompt",
        "rag",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "neural",
        "network",
        "deep",
    }

    texts = []
    for d in docs:
        toks = [t.lower() for t in tokenize_simple(d)]
        toks = [t for t in toks if len(t) >= 2 and (t not in stop) and (not t.isdigit())]
        texts.append(toks)

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_tokens=list(keep_tokens))
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        chunksize=chunksize,
        random_state=0,
        alpha="auto",
        eta="auto",
    )

    doc_topics = [lda.get_document_topics(bow, minimum_probability=0.0) for bow in corpus]
    assigned = [max(dt, key=lambda x: x[1])[0] if dt else -1 for dt in doc_topics]

    topic_keywords: dict[int, list[str]] = {}
    for tid in range(num_topics):
        topic_keywords[tid] = [w for w, _ in lda.show_topic(tid, topn=20)]

    model_dir.mkdir(parents=True, exist_ok=True)
    dump({"lda": lda, "dictionary": dictionary}, model_dir / "lda_model.joblib")
    pd.DataFrame(
        [{"topic_id": k, "keywords": ", ".join(v)} for k, v in sorted(topic_keywords.items())]
    ).to_csv(model_dir / "lda_topics.csv", index=False)

    logger.info(f"LDA fitted with {num_topics:,} topics")
    return assigned, doc_topics, topic_keywords


def compute_topic_features(
    df: pd.DataFrame,
    *,
    text_col: str = "clean_transcript",
    model_dir: Path,
    method: str,
    lda_num_topics: int,
    lda_passes: int,
    lda_chunksize: int,
    logger,
) -> TopicResult:
    docs = df[text_col].fillna("").tolist()

    use_method = method
    if method == "auto":
        try:
            import bertopic

            use_method = "bertopic"
        except Exception:
            use_method = "lda"

    if use_method == "bertopic":
        try:
            topics, probs, topic_keywords = _fit_transform_bertopic(docs, model_dir=model_dir, logger=logger)
            ai_topics = _ai_topics_from_keywords(topic_keywords)
            if not ai_topics:
                if "ani_kw_per1k" in df.columns:
                    ai_score = pd.to_numeric(df["ani_kw_per1k"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
                elif "ani_any" in df.columns:
                    ai_score = pd.to_numeric(df["ani_any"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
                else:
                    ai_score = _doc_ai_score_from_text(docs)
                ai_topics = _select_ai_topics_by_enrichment(topics, ai_score, logger=logger)
            features = pd.DataFrame({"topic_id": pd.Series(topics, dtype="int32")})

            if probs is None:
                features["ai_topic_share"] = features["topic_id"].isin(ai_topics).astype(float)
            else:
                probs = np.asarray(probs)
                topic_ids = sorted([tid for tid in topic_keywords.keys() if tid != -1])
                if probs.ndim == 2 and probs.shape[1] == len(topic_ids):
                    idx = [i for i, tid in enumerate(topic_ids) if tid in ai_topics]
                    features["ai_topic_share"] = probs[:, idx].sum(axis=1) if idx else 0.0
                else:
                    features["ai_topic_share"] = features["topic_id"].isin(ai_topics).astype(float)

            features["ai_topic_share"] = features["ai_topic_share"].astype(float).where(features["ai_topic_share"] >= 1e-6, 0.0)
            return TopicResult(features=features, ai_topic_ids=ai_topics, method="bertopic")
        except Exception as e:
            logger.info(f"BERTopic failed ({e}); falling back to LDA")
            use_method = "lda"

    if use_method == "lda":
        try:
            assigned, doc_topics, topic_keywords = _fit_transform_lda(
                docs,
                num_topics=lda_num_topics,
                passes=lda_passes,
                chunksize=lda_chunksize,
                model_dir=model_dir,
                logger=logger,
            )
            ai_topics = _ai_topics_from_keywords(topic_keywords)
            if not ai_topics:
                if "ani_kw_per1k" in df.columns:
                    ai_score = pd.to_numeric(df["ani_kw_per1k"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
                elif "ani_any" in df.columns:
                    ai_score = pd.to_numeric(df["ani_any"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
                else:
                    ai_score = _doc_ai_score_from_text(docs)
                ai_topics = _select_ai_topics_by_enrichment(assigned, ai_score, logger=logger)
            ai_share = []
            for dt in doc_topics:
                s = 0.0
                for tid, p in dt:
                    if int(tid) in ai_topics:
                        s += float(p)
                ai_share.append(s)

            features = pd.DataFrame(
                {
                    "topic_id": pd.Series(assigned, dtype="int32"),
                    "ai_topic_share": pd.Series(ai_share, dtype="float64"),
                }
            )
            return TopicResult(features=features, ai_topic_ids=ai_topics, method="lda")
        except Exception as e:
            logger.info(f"LDA failed ({e}); emitting zero topic features")

    features = pd.DataFrame(
        {
            "topic_id": pd.Series([-1] * len(df), dtype="int32"),
            "ai_topic_share": pd.Series([0.0] * len(df), dtype="float64"),
        }
    )
    return TopicResult(features=features, ai_topic_ids=set(), method="none")
