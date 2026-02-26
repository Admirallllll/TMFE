# Benchmark Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible benchmark comparison pipeline for predicting `ai_initiation_score` with shared CV splits and exportable slide-ready tables/plots.

**Architecture:** Add a dedicated `src/analysis/benchmark_comparison.py` module that evaluates mean baseline, metadata-only linear model(s), and text-only Lasso under identical `GroupKFold(by ticker)` splits. Reuse existing feature outputs (`regression_dataset.parquet`, `sentences_with_keywords.parquet`) and expose one `run_benchmark_comparison(...)` entrypoint for CLI/pipeline use.

**Tech Stack:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `pytest`

### Task 1: Add failing benchmark output-format tests

**Files:**
- Create: `tests/test_benchmark_comparison.py`
- Test: `tests/test_benchmark_comparison.py`

**Step 1: Write the failing test**

```python
def test_benchmark_summary_is_flat_table_and_models_differ():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_comparison.py -v`
Expected: FAIL because module/function does not exist yet

**Step 3: Write minimal implementation**

Create benchmark module with evaluation + CSV summary export helpers.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_comparison.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_benchmark_comparison.py src/analysis/benchmark_comparison.py
git commit -m "feat: add reproducible benchmark comparison evaluation"
```

### Task 2: Integrate benchmark stage into pipeline

**Files:**
- Modify: `run_pipeline.py`
- Test: `tests/test_benchmark_comparison.py`

**Step 1: Write the failing test**

Add/update test for callable entrypoint and stable output schema if needed.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark_comparison.py -v`
Expected: FAIL for missing integration behavior/schema

**Step 3: Write minimal implementation**

Add Stage 10.5/11 benchmark call (after regression + optional lasso) with CLI flags for `--skip-benchmark`, `--benchmark-cv-folds`, `--benchmark-group-col`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark_comparison.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add run_pipeline.py tests/test_benchmark_comparison.py
git commit -m "feat: wire benchmark comparison into pipeline"
```

### Task 3: Verify on real outputs

**Files:**
- Read: `outputs/features/regression_dataset.parquet`
- Read: `outputs/features/sentences_with_keywords.parquet`
- Write: `outputs/figures/benchmark_comparison_folds.csv`
- Write: `outputs/figures/benchmark_comparison_summary.csv`
- Write: `outputs/figures/benchmark_comparison.png`

**Step 1: Run benchmark module**

Run: `python -m src.analysis.benchmark_comparison --regression-dataset ... --sentences ...`
Expected: Generates non-identical model metrics and flat CSV summary table.

**Step 2: Sanity-check outputs**

Run: `python - <<'PY' ...`
Expected: `Kendall Tau` and `MAE` columns numeric; no duplicate header row; at least one model differs from mean baseline.

**Step 3: Run tests**

Run: `pytest tests/test_benchmark_comparison.py tests/test_regression.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add outputs/figures/benchmark_comparison_* tests/test_benchmark_comparison.py
git commit -m "fix: regenerate benchmark comparison outputs with shared cv splits"
```
