import pandas as pd

from src.research.data import build_research_dataset, run_basic_sanity_checks


def test_build_research_dataset_has_required_targets_and_keys():
    doc_metrics = pd.DataFrame(
        {
            "doc_id": ["AAA_2024Q1", "AAA_2024Q2", "BBB_2024Q1"],
            "speech_kw_ai_ratio": [0.1, 0.2, 0.0],
            "qa_kw_ai_ratio": [0.05, 0.10, 0.0],
            "overall_kw_ai_ratio": [0.07, 0.14, 0.0],
        }
    )
    initiation = pd.DataFrame(
        {
            "doc_id": ["AAA_2024Q1", "AAA_2024Q2", "BBB_2024Q1"],
            "total_ai_exchanges": [2, 3, 0],
            "ai_initiation_score": [0.4, 0.7, 0.5],
            "analyst_initiated_ratio": [0.6, 0.3, 0.0],
            "management_pivot_ratio": [0.4, 0.7, 0.0],
        }
    )
    sentences = pd.DataFrame(
        {
            "doc_id": ["AAA_2024Q1", "AAA_2024Q1", "AAA_2024Q2", "BBB_2024Q1"],
            "section": ["qa", "speech", "qa", "qa"],
            "role": ["analyst", "management", "management", "analyst"],
            "turn_idx": [0, 0, 1, 0],
            "kw_is_ai": [True, False, True, False],
            "text": ["ai demand", "prepared remarks", "ai roadmap", "no ai"],
        }
    )
    parsed = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "year": [2024, 2024, 2024],
            "quarter": [1, 2, 1],
            "speech_word_count": [100, 90, 80],
            "qa_word_count": [200, 180, 120],
            "num_qa_exchanges": [10, 9, 8],
            "date": ["2024-03-01", "2024-06-01", "2024-03-02"],
        }
    )
    final_dataset = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "year": [2024, 2024, 2024],
            "quarter": [1, 2, 1],
            "sector": ["Tech", "Tech", "Health"],
            "industry": ["Software", "Software", "Biotech"],
            "industry_name": ["Software", "Software", "Biotech"],
            "gsector": [45, 45, 35],
            "gsubind": [4510, 4510, 3520],
            "date": ["2024-03-01", "2024-06-01", "2024-03-02"],
        }
    )
    wrds = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "AAA"],
            "year": [2024, 2024, 2024, 2024],
            "quarter": [1, 2, 1, 3],
            "gsector": [45, 45, 35, 45],
            "mkvaltq": [1000, 1100, 900, 1200],
            "xrdq": [50, 55, 0, 60],
            "prccq": [100, 110, 50, 120],
            "epspxq": [1.0, 1.1, 0.5, 1.2],
            "cshoq": [10, 10, 18, 10],
            "rd_intensity": [0.05, 0.05, 0.0, 0.05],
            "log_mktcap": [6.90, 7.00, 6.80, 7.09],
            "ln_price": [4.60, 4.70, 3.91, 4.79],
            "eps_positive": [1.0, 1.0, 1.0, 1.0],
            "eps_growth_yoy": [0.1, 0.1, 0.0, 0.1],
            "price_growth_yoy": [0.2, 0.2, 0.0, 0.2],
            "mktcap_growth_qoq": [0.1, 0.1, 0.0, 0.1],
            "rd_intensity_change_qoq": [0.0, 0.0, 0.0, 0.0],
            "y_next_rd_intensity_change": [0.0, 0.0, 0.0, 0.0],
            "y_next_mktcap_growth": [0.1, 0.09, 0.0, 0.0],
            "y_next_eps_growth_yoy": [0.2, 0.1, 0.0, 0.0],
        }
    )

    result = build_research_dataset(
        document_metrics=doc_metrics,
        initiation_scores=initiation,
        sentences_with_keywords=sentences,
        parsed_transcripts=parsed,
        final_dataset=final_dataset,
        wrds_features=wrds,
    )

    ds = result.dataset
    assert set(["doc_id", "ticker", "year", "quarter", "quarter_index"]).issubset(ds.columns)
    assert "y_next_mktcap_growth" in ds.columns
    assert len(ds) == 3

    run_basic_sanity_checks(ds)
