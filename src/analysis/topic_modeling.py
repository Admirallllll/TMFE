"""
Topic Modeling Module (Quarterly)

Uses LDA to extract topics per quarter. Designed for manual topic naming.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import os
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation


def _parse_doc_id(doc_id: str) -> Tuple[Optional[int], Optional[int]]:
    parts = str(doc_id).rsplit("_", 1)
    if len(parts) != 2:
        return None, None
    yq = parts[1]
    if "Q" not in yq:
        return None, None
    try:
        year = int(yq.split("Q")[0])
        quarter = int(yq.split("Q")[1])
        return year, quarter
    except Exception:
        return None, None


def _build_stopwords() -> List[str]:
    custom = {
        "company", "companies", "quarter", "year", "management", "analyst",
        "call", "calls", "question", "questions", "answer", "answers",
        "thank", "thanks", "good", "morning", "afternoon", "evening",
        "said", "say", "says", "will", "would", "could", "should",
        "also", "one", "two", "three", "four", "five",
        "customers", "customer", "business", "businesses"
    }
    # CountVectorizer expects 'english', list, or None (not a set).
    return sorted(set(ENGLISH_STOP_WORDS).union(custom))


def _prepare_docs(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    doc_text = df.groupby("doc_id")["text"].apply(lambda x: " ".join(x.astype(str))).reset_index()
    return doc_text["doc_id"].tolist(), doc_text["text"].tolist()


def _extract_topics(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n_words: int = 12
) -> List[Dict]:
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_idx = np.argsort(topic_weights)[::-1][:top_n_words]
        top_terms = [feature_names[i] for i in top_idx]
        top_weights = [float(topic_weights[i]) for i in top_idx]
        topics.append({
            "topic_id": int(topic_idx),
            "topic_label": "",
            "top_terms": " | ".join(top_terms),
            "top_weights": " | ".join([f"{w:.4f}" for w in top_weights])
        })
    return topics


def run_quarterly_topic_modeling(
    sentences_path: str,
    output_dir: str = "outputs/features",
    start_year: int = 2020,
    end_year: int = 2025,
    n_topics: int = 20,
    top_n_words: int = 12,
    filter_ai: bool = True,
    min_docs: int = 10,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run LDA per quarter and save topic tables for manual naming.

    If filter_ai=True, uses kw_is_ai==1 sentences to focus on AI topics.
    """
    topics_dir = os.path.join(output_dir, "topics")
    os.makedirs(topics_dir, exist_ok=True)

    print("Loading sentences for topic modeling...")
    cols = ["text", "doc_id"]
    if filter_ai:
        cols.append("kw_is_ai")
    df = pd.read_parquet(sentences_path, columns=cols)
    if filter_ai:
        df = df[df["kw_is_ai"] == 1].copy()

    if len(df) == 0:
        print("No sentences available for topic modeling. Skipping.")
        return pd.DataFrame()

    df["year"], df["quarter"] = zip(*df["doc_id"].map(_parse_doc_id))
    df = df.dropna(subset=["year", "quarter"])
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)

    stopwords = _build_stopwords()
    all_topics: List[Dict] = []

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            q_df = df[(df["year"] == year) & (df["quarter"] == quarter)]
            if len(q_df) == 0:
                continue

            doc_ids, docs = _prepare_docs(q_df)
            if len(docs) < min_docs:
                print(f"Skipping {year}Q{quarter}: not enough documents ({len(docs)}).")
                continue

            vec = CountVectorizer(
                stop_words=stopwords,
                max_df=0.9,
                min_df=2,
                max_features=max_features,
                ngram_range=ngram_range
            )
            X = vec.fit_transform(docs)

            if X.shape[1] == 0:
                print(f"Skipping {year}Q{quarter}: no features after vectorization.")
                continue

            n_components = min(n_topics, len(docs), X.shape[1])
            if n_components < 2:
                print(f"Skipping {year}Q{quarter}: insufficient components.")
                continue

            lda = LatentDirichletAllocation(
                n_components=n_components,
                random_state=random_state,
                learning_method="batch"
            )
            doc_topic = lda.fit_transform(X)

            topics = _extract_topics(lda, vec, top_n_words=top_n_words)
            for t in topics:
                t.update({
                    "year": year,
                    "quarter": quarter,
                    "n_docs": len(docs),
                    "n_terms": int(X.shape[1])
                })
                all_topics.append(t)

            # Save per-quarter topic table
            q_topics_df = pd.DataFrame(topics)
            q_topics_path = os.path.join(topics_dir, f"topics_{year}Q{quarter}.csv")
            q_topics_df.to_csv(q_topics_path, index=False)

            # Save doc-topic distributions for manual inspection
            doc_topic_df = pd.DataFrame(doc_topic, columns=[f"topic_{i}" for i in range(doc_topic.shape[1])])
            doc_topic_df.insert(0, "doc_id", doc_ids)
            doc_topic_df["dominant_topic"] = doc_topic_df.drop(columns=["doc_id"]).idxmax(axis=1)
            doc_topic_path = os.path.join(topics_dir, f"doc_topics_{year}Q{quarter}.parquet")
            doc_topic_df.to_parquet(doc_topic_path, index=False)

            print(f"Saved topics for {year}Q{quarter}: {q_topics_path}")

    topics_df = pd.DataFrame(all_topics)
    summary_path = os.path.join(topics_dir, "topics_per_quarter.csv")
    topics_df.to_csv(summary_path, index=False)

    # Save manifest for manual naming workflow
    manifest = {
        "filter_ai": filter_ai,
        "n_topics": n_topics,
        "top_n_words": top_n_words,
        "min_docs": min_docs,
        "max_features": max_features,
        "ngram_range": ngram_range,
        "output_dir": topics_dir
    }
    with open(os.path.join(topics_dir, "topic_model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved topic summary to {summary_path}")
    return topics_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quarterly topic modeling (LDA)")
    parser.add_argument("--sentences", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/features")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--n-topics", type=int, default=20)
    parser.add_argument("--top-words", type=int, default=12)
    parser.add_argument("--filter-ai", action="store_true", help="Use only kw_is_ai sentences")

    args = parser.parse_args()

    run_quarterly_topic_modeling(
        args.sentences,
        args.output_dir,
        args.start_year,
        args.end_year,
        n_topics=args.n_topics,
        top_n_words=args.top_words,
        filter_ai=args.filter_ai
    )
