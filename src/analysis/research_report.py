"""Research-grade reporting stage for AI narrative + WRDS analysis."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from src.research.data import (
    build_research_dataset,
    prepare_wrds_features,
    run_basic_sanity_checks,
)
from src.research.models import (
    build_deep_dive_cases,
    run_fe_regressions,
    run_interpretable_lasso,
    run_model_comparison,
)
from src.research.report import write_report
from src.research.viz import (
    plot_dataset_overview,
    plot_lasso_outputs,
    plot_metadata_association,
    plot_model_comparison,
    plot_quadrants,
    plot_structural_metadata,
    plot_time_series,
)


def run_research_report(
    sentences_with_keywords_path: str,
    document_metrics_path: str,
    initiation_scores_path: str,
    parsed_transcripts_path: str,
    final_dataset_path: str,
    wrds_path: str,
    output_dir: str = "outputs/report",
    model_target: str = "y_next_mktcap_growth",
    test_quarters: int = 4,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    tab_dir = os.path.join(output_dir, "tables")
    lasso_dir = os.path.join(output_dir, "lasso")
    case_dir = os.path.join(output_dir, "cases")
    for d in [fig_dir, tab_dir, lasso_dir, case_dir]:
        os.makedirs(d, exist_ok=True)

    sentences_kw = pd.read_parquet(sentences_with_keywords_path)
    doc_metrics = pd.read_parquet(document_metrics_path)
    initiation = pd.read_parquet(initiation_scores_path) if os.path.exists(initiation_scores_path) else pd.DataFrame()
    parsed = pd.read_parquet(parsed_transcripts_path) if os.path.exists(parsed_transcripts_path) else pd.DataFrame()
    final_dataset = pd.read_parquet(final_dataset_path) if os.path.exists(final_dataset_path) else pd.DataFrame()

    wrds = prepare_wrds_features(wrds_path)

    build = build_research_dataset(
        document_metrics=doc_metrics,
        initiation_scores=initiation,
        sentences_with_keywords=sentences_kw,
        parsed_transcripts=parsed,
        final_dataset=final_dataset,
        wrds_features=wrds,
    )

    research_df = build.dataset
    data_dict = build.data_dictionary

    run_basic_sanity_checks(research_df)

    research_dataset_path = os.path.join("outputs", "features", "research_dataset.parquet")
    data_dict_path = os.path.join("outputs", "features", "data_dictionary.csv")
    os.makedirs(os.path.dirname(research_dataset_path), exist_ok=True)
    research_df.to_parquet(research_dataset_path, index=False)
    data_dict.to_csv(data_dict_path, index=False)

    figure_notes: List[Dict[str, str]] = []

    note = plot_dataset_overview(research_df, os.path.join(fig_dir, "dataset_overview.png"))
    figure_notes.append(note)

    assoc_df, note = plot_metadata_association(research_df, os.path.join(fig_dir, "metadata_association.png"))
    assoc_df.to_csv(os.path.join(tab_dir, "metadata_association.csv"), index=False)
    figure_notes.append(note)

    note = plot_structural_metadata(research_df, os.path.join(fig_dir, "structural_metadata.png"))
    figure_notes.append(note)

    note = plot_time_series(research_df, os.path.join(fig_dir, "ai_time_series.png"))
    figure_notes.append(note)

    quadrant_df, note = plot_quadrants(
        research_df,
        os.path.join(fig_dir, "company_quadrants_research.png"),
        os.path.join(tab_dir, "quadrant_representatives.csv"),
    )
    quadrant_df.to_csv(os.path.join(tab_dir, "quadrants_all.csv"), index=False)
    figure_notes.append(note)

    fe_summary = run_fe_regressions(research_df, output_dir=tab_dir)

    if model_target not in research_df.columns:
        # Fallback to available next-period target.
        for candidate in ["y_next_eps_growth_yoy", "y_next_rd_intensity_change"]:
            if candidate in research_df.columns:
                model_target = candidate
                break

    comparison = run_model_comparison(
        research_df,
        target=model_target,
        output_dir=tab_dir,
        test_quarters=test_quarters,
    )
    note = plot_model_comparison(comparison.summary, os.path.join(fig_dir, "model_comparison.png"))
    figure_notes.append(note)

    lasso_outputs = run_interpretable_lasso(
        dataset=research_df,
        sentences_df=sentences_kw,
        output_dir=lasso_dir,
        target=model_target,
        section="qa",
        test_quarters=test_quarters,
    )
    lasso_notes = plot_lasso_outputs(
        term_df=lasso_outputs["terms"],
        stability_df=lasso_outputs["stability"],
        output_dir=fig_dir,
    )
    figure_notes.extend(lasso_notes)

    deep_dive = build_deep_dive_cases(
        dataset=research_df,
        lasso_predictions=lasso_outputs["predictions"],
        sentences_df=sentences_kw,
        output_path=os.path.join(case_dir, "deep_dive_cases.csv"),
    )

    notes_df = pd.DataFrame(figure_notes)
    notes_df.to_csv(os.path.join(tab_dir, "figure_takeaways.csv"), index=False)

    report_path = os.path.join(output_dir, "report.md")
    write_report(
        report_path=report_path,
        dataset=research_df,
        data_dictionary_path=data_dict_path,
        figure_notes=figure_notes,
        fe_summary=fe_summary,
        model_comparison=comparison.summary,
        lasso_metrics=lasso_outputs["metrics"],
        lasso_top_terms=lasso_outputs["top_terms"],
        deep_dive_cases=deep_dive,
    )

    return {
        "research_dataset_path": research_dataset_path,
        "data_dictionary_path": data_dict_path,
        "report_path": report_path,
        "figure_dir": fig_dir,
        "table_dir": tab_dir,
        "lasso_dir": lasso_dir,
        "case_dir": case_dir,
    }


if __name__ == "__main__":
    out = run_research_report(
        sentences_with_keywords_path="outputs/features/sentences_with_keywords.parquet",
        document_metrics_path="outputs/features/document_metrics.parquet",
        initiation_scores_path="outputs/features/initiation_scores.parquet",
        parsed_transcripts_path="outputs/features/parsed_transcripts.parquet",
        final_dataset_path="final_dataset.parquet",
        wrds_path="Sp500_meta_data.csv",
        output_dir="outputs/report",
    )
    print("Research report outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")
