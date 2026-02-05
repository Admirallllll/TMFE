# QA Transcript Pipeline Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the prediction-focused pipeline with a new Q&A descriptive pipeline that parses prepared remarks/Q&A, builds call/turn/QA-pair tables, detects AI mentions, and generates validated trend analyses with fail-fast checks.

**Architecture:** Add a new QA pipeline entrypoint (`run_all_qa.py`) and supporting modules for parsing, AI detection, analysis, visualization, and validation. Quarantine legacy prediction code under `src/legacy/` and keep shared utilities. All outputs are cached as parquet and validated before plots/tables are generated.

**Tech Stack:** Python 3.10+, pandas, datasets, regex, (optional) BERTopic or embeddings + k-means, existing transfer encoder.

---

### Task 1: Define QA config constants + paths (thresholds, QC) and tests

**Files:**
- Modify: `src/config.py`
- Create: `tests/test_qa_config.py`

**Step 1: Write the failing test**
```python
from src.config import QAConfig

def test_qa_config_constants():
    cfg = QAConfig()
    assert cfg.min_qa_found_rate_dev == 0.60
    assert cfg.min_qa_found_rate_full == 0.80
    assert cfg.min_assigned_char_pct_dev == 0.70
    assert cfg.min_assigned_char_pct_full == 0.80
    assert cfg.ai_thr_lo <= cfg.ai_thr_hi
```

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_qa_config.py -v`
Expected: FAIL with `ImportError` or missing attributes

**Step 3: Write minimal implementation**
```python
@dataclass(frozen=True)
class QAConfig:
    min_qa_found_rate_dev: float = 0.60
    min_qa_found_rate_full: float = 0.80
    min_assigned_char_pct_dev: float = 0.70
    min_assigned_char_pct_full: float = 0.80
    ai_thr_hi: float = 0.80
    ai_thr_lo: float = 0.65
    def __post_init__(self):
        object.__setattr__(self, "ai_thr_lo", min(self.ai_thr_lo, self.ai_thr_hi))
```

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_qa_config.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/config.py tests/test_qa_config.py
git commit -m "feat: add QA config constants and thresholds"
```

---

### Task 2: Implement transcript parsing + roster matching

**Files:**
- Create: `src/data/parse_transcripts.py`
- Create: `tests/test_parser_split.py`
- Create: `tests/test_roster_matching.py`

**Step 1: Write failing tests**
```python
from src.data.parse_transcripts import split_prepared_and_qa, extract_rosters

def test_split_prepared_and_qa():
    txt = "Prepared Remarks\n...\nQuestion-and-Answer Session\nOPERATOR: ..."
    prepared, qa, diag = split_prepared_and_qa(txt)
    assert diag["qa_header_matched"] is True
    assert diag["qa_start_idx"] >= 0
    assert len(prepared) > 0 and len(qa) > 0


def test_extract_rosters():
    txt = "CORPORATE PARTICIPANTS\nJohn Doe - CEO\nANALYSTS\nJane Smith - Big Bank"
    roster = extract_rosters(txt)
    assert roster.get("john doe") == "management"
    assert roster.get("jane smith") == "analyst"
```

**Step 2: Run tests to verify fail**
Run: `pytest tests/test_parser_split.py tests/test_roster_matching.py -v`
Expected: FAIL (module not found)

**Step 3: Implement minimal parsing & roster normalization**
- Regex headers for Q&A sections
- Extract roster blocks and normalize names (lower, strip punctuation, remove middle names, last-name fallback)

**Step 4: Run tests to verify pass**
Run: `pytest tests/test_parser_split.py tests/test_roster_matching.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/data/parse_transcripts.py tests/test_parser_split.py tests/test_roster_matching.py
git commit -m "feat: add transcript split and roster parsing"
```

---

### Task 3: Implement turn extraction + state machine + tests

**Files:**
- Modify: `src/data/parse_transcripts.py`
- Create: `tests/test_turn_state_machine.py`

**Step 1: Write failing test**
```python
from src.data.parse_transcripts import extract_turns

def test_turn_state_machine():
    qa = "ANALYST: What about AI?\nCEO: We invest.\nCFO: And cost savings.\nANALYST: Follow-up.\nOPERATOR: Next question.\nCEO: More AI."
    turns, diag = extract_turns(qa, roster={})
    q_ids = [t.question_id for t in turns if t.turn_type == "question"]
    assert q_ids == [1, 2]
```

**Step 2: Run test to verify fail**
Run: `pytest tests/test_turn_state_machine.py -v`
Expected: FAIL

**Step 3: Implement state machine**
- Analyst starts new `question_id`
- Management turns share `answer_group_id` until next analyst
- Operator does not start new question
- Turn struct includes `speaker_raw_header`, `char_start/end`, `turn_text_raw/clean`

**Step 4: Run test to verify pass**
Run: `pytest tests/test_turn_state_machine.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/data/parse_transcripts.py tests/test_turn_state_machine.py
git commit -m "feat: add turn extraction state machine"
```

---

### Task 4: Build calls/turns/qa_pairs tables + call_id uniqueness

**Files:**
- Create: `src/data/build_calls.py`
- Create: `src/data/build_turns.py`
- Create: `src/data/build_qa_pairs.py`
- Create: `tests/test_call_id_uniqueness.py`

**Step 1: Write failing test**
```python
from src.data.build_calls import build_calls_table

def test_call_id_unique():
    df = ...  # minimal dataframe with duplicate ticker/datacqtr
    calls = build_calls_table(df)
    assert calls.call_id.is_unique
```

**Step 2: Run test to verify fail**
Run: `pytest tests/test_call_id_uniqueness.py -v`
Expected: FAIL

**Step 3: Implement table builders**
- `call_id = ticker|datacqtr|earnings_date` with hash fallback
- Persist parse diagnostics and roster coverage metrics in calls

**Step 4: Run test to verify pass**
Run: `pytest tests/test_call_id_uniqueness.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/data/build_calls.py src/data/build_turns.py src/data/build_qa_pairs.py tests/test_call_id_uniqueness.py
git commit -m "feat: build calls/turns/qa_pairs tables"
```

---

### Task 5: AI detection + calibration utilities + tests

**Files:**
- Create: `src/features/ai_detection.py`
- Create: `tests/test_ai_detection.py`

**Step 1: Write failing tests**
```python
from src.features.ai_detection import apply_ai_rules

def test_ai_rule_thresholds():
    row = {"is_ai_kw": False, "ai_score_encoder": 0.9}
    assert apply_ai_rules(row, thr_hi=0.8, thr_lo=0.6)["is_ai_final"] is True
```

**Step 2: Run test to verify fail**
Run: `pytest tests/test_ai_detection.py -v`
Expected: FAIL

**Step 3: Implement AI rule + calibration**
- `is_ai_final = is_ai_kw OR ai_score_encoder >= thr_hi`
- `needs_review = thr_lo <= score < thr_hi`
- Enforce `thr_lo <= thr_hi` in config and log thresholds

**Step 4: Run test to verify pass**
Run: `pytest tests/test_ai_detection.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/features/ai_detection.py tests/test_ai_detection.py
git commit -m "feat: add AI detection rules and calibration utilities"
```

---

### Task 6: Analysis modules + core tables

**Files:**
- Create: `src/analysis/trends.py`
- Create: `src/analysis/introducer.py`
- Create: `src/analysis/uncertainty.py`

**Step 1: Write failing tests**
```python
from src.analysis.trends import ai_question_rate_by_quarter

def test_ai_question_rate_nonempty():
    df = ...  # minimal turns df
    out = ai_question_rate_by_quarter(df)
    assert len(out) > 0
```

**Step 2: Run test to verify fail**
Run: `pytest tests/test_analysis_trends.py -v`
Expected: FAIL

**Step 3: Implement analysis functions**
- Output CSV-ready tables for trends, introducer, uncertainty

**Step 4: Run tests to verify pass**
Run: `pytest tests/test_analysis_trends.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/analysis/*.py tests/test_analysis_trends.py
git commit -m "feat: add QA trend/introducer/uncertainty analysis"
```

---

### Task 7: Visualization + optional topic clustering guard

**Files:**
- Create: `src/viz/qa_figures.py`
- Create: `src/analysis/topics.py` (optional)

**Step 1: Write failing test**
```python
from src.viz.qa_figures import plot_trend

def test_plot_raises_on_empty():
    import pandas as pd
    try:
        plot_trend(pd.DataFrame())
        assert False
    except ValueError:
        assert True
```

**Step 2: Run test to verify fail**
Run: `pytest tests/test_viz_empty.py -v`
Expected: FAIL

**Step 3: Implement plots + empty checks**
- If optional topic table is empty: write RESULTS.md entry explaining skip OR generate placeholder plot

**Step 4: Run test to verify pass**
Run: `pytest tests/test_viz_empty.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/viz/qa_figures.py src/analysis/topics.py tests/test_viz_empty.py
git commit -m "feat: add QA plots and empty-table guards"
```

---

### Task 8: Pipeline entrypoint + validation

**Files:**
- Create: `src/pipeline/run_all_qa.py`
- Modify: `src/pipeline/validate_outputs.py`
- Create: `tests/test_validation.py`

**Step 1: Write failing test**
```python
from src.pipeline.validate_outputs import validate_tables

def test_validation_fails_empty():
    import pandas as pd
    try:
        validate_tables(pd.DataFrame())
        assert False
    except ValueError:
        assert True
```

**Step 2: Run test to verify fail**
Run: `pytest tests/test_validation.py -v`
Expected: FAIL

**Step 3: Implement pipeline + validation**
- DEV warn-only for threshold gates; structural empties always raise
- Log shape/null/head before each core plot

**Step 4: Run test to verify pass**
Run: `pytest tests/test_validation.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add src/pipeline/run_all_qa.py src/pipeline/validate_outputs.py tests/test_validation.py
git commit -m "feat: add QA pipeline entrypoint and fail-fast validation"
```

---

### Task 9: Legacy quarantine + docs

**Files:**
- Move: `src/models/` -> `src/legacy/models/`
- Move: `src/viz/results_*.py` -> `src/legacy/viz/`
- Modify: `README.md`
- Create: `docs/REFACTOR_REVIEW.md`

**Step 1: Write doc outline**
- Summarize changes, link to new outputs, list figures

**Step 2: Update README**
- Replace prediction objective with QA trend objective, new run command

**Step 3: Move legacy code**
- Update imports if needed

**Step 4: Commit**
```bash
git add README.md docs/REFACTOR_REVIEW.md src/legacy/
git commit -m "docs: update README and quarantine legacy pipeline"
```

---

### Task 10: DEV run + FULL run

**Run DEV:**
```
python -m src.pipeline.run_all_qa --dev --dev-sample-n 200 --recompute
```
Expected: generates non-empty tables/figures in dev folders; logs warnings if threshold gates not met.

**Run FULL:**
```
python -m src.pipeline.run_all_qa --recompute
```
Expected: fail-fast if any core tables/figures empty; outputs tables/figures + RESULTS.md.

**Commit**
```bash
git add outputs/ logs/ data/processed/ docs/RESULTS.md
git commit -m "chore: run QA pipeline dev/full"
```
