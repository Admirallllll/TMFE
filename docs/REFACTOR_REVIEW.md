# QA Pipeline Refactor Review

## Summary of Changes
- Replaced prediction-focused pipeline with a Q&A descriptive pipeline.
- Added transcript parsing: prepared remarks vs Q&A split, turn-level parsing, and QA-pair construction.
- Implemented deterministic AI detection with keyword + encoder rules and calibration utilities.
- Added trend analyses and validation checks that fail fast on empty or invalid outputs.
- Quarantined legacy prediction code under `src/legacy/`.

## Alignment with Professor Feedback
- **Speech vs Q&A separation**: Prepared remarks and Q&A are split and analyzed separately.
- **Turn-level Q&A**: One row per turn with question/answer types and speaker roles.
- **QA-pairs**: One row per question with 1..N answers, preserving multi-answer handling.
- **Focus on descriptive trends**: No forecasting; outputs are rates/volumes and introducer patterns.

## Key Outputs (Files)
- `data/processed/calls.parquet`
- `data/processed/turns.parquet`
- `data/processed/qa_pairs.parquet`
- `outputs/tables/*.csv`
- `outputs/figures/*.png`

## Figures
- `trend_ai_question_rate_by_quarter.png`
- `trend_ai_answer_rate_by_quarter.png`
- `who_introduces_ai_first_by_quarter.png`
- `who_introduces_ai_first_by_sector.png`
- `ai_first_turn_position_distribution.png`
- `answers_uncertainty_ai_vs_nonai.png`
- `answers_uncertainty_analystfirst_vs_mgmtfirst.png`
- Optional: `ai_question_subtopics_trend.png` (placeholder if skipped)

## Notes
- Validation will fail if any core table or figure input is empty or all NaN.
- DEV mode warns on parse thresholds; FULL mode fails fast.
