import pandas as pd

from src.analysis.topic_modeling import run_quarterly_topic_modeling


def _make_sentences(n_docs: int, year: int = 2024, quarter: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(n_docs):
        doc_id = f"TK{i:03d}_{year}Q{quarter}"
        grp = i % 4
        for j in range(3):
            rows.append(
                {
                    "doc_id": doc_id,
                    "text": f"ai platform innovation cluster{grp} feature{grp} market demand {j}",
                    "kw_is_ai": 1,
                }
            )
    return pd.DataFrame(rows)


def test_topic_modeling_generates_cluster_plot(tmp_path):
    sentences = _make_sentences(12)
    sentences_path = tmp_path / "sentences.parquet"
    out_dir = tmp_path / "features"
    sentences.to_parquet(sentences_path, index=False)

    run_quarterly_topic_modeling(
        sentences_path=str(sentences_path),
        output_dir=str(out_dir),
        start_year=2024,
        end_year=2024,
        n_topics=4,
        top_n_words=5,
        filter_ai=True,
        min_docs=5,
        max_features=100,
        ngram_range=(1, 1),
        generate_cluster_plots=True,
    )

    assert (out_dir / "topics" / "topic_cluster_2024Q1.png").exists()
    assert (out_dir / "topics" / "doc_topics_2024Q1.parquet").exists()


def test_topic_modeling_small_data_skips_cleanly(tmp_path):
    sentences = _make_sentences(3)
    sentences_path = tmp_path / "sentences_small.parquet"
    out_dir = tmp_path / "features_small"
    sentences.to_parquet(sentences_path, index=False)

    result = run_quarterly_topic_modeling(
        sentences_path=str(sentences_path),
        output_dir=str(out_dir),
        start_year=2024,
        end_year=2024,
        n_topics=3,
        top_n_words=5,
        filter_ai=True,
        min_docs=10,
        max_features=100,
        ngram_range=(1, 1),
        generate_cluster_plots=True,
    )

    assert isinstance(result, pd.DataFrame)
