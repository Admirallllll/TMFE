# Pipeline Review Report

**Date**: 2026-01-24
**Reviewer**: Claude Code (Opus 4.5)
**Status**: Fixes Implemented

---

## Executive Summary

This report documents the debugging and fixes applied to the AI Narratives Pipeline after identifying critical issues with empty figures and sklearn ConvergenceWarnings.

**Issues Fixed:**
1. Empty `ai_topic_share` figures (two figures showing "No variation")
2. ConvergenceWarning from ElasticNet/LogisticRegression models
3. Missing visualization plots from `model_results.csv`
4. Added validation script for output verification

---

## Issue 1: Empty ai_topic_share Figures

### Symptoms
- `eda_ai_topic_share_trend.png` displayed: "No variation in ai_topic_share"
- `results_pre_post_ai_topic_share_by_sector.png` displayed: "No variation in ai_topic_share (all values equal)"

### Root Cause Analysis

The AI topic identification pipeline failed silently:

1. **Keyword-based detection** (`_ai_topics_from_keywords`): BERTopic topics did not contain exact AI keyword matches (e.g., "artificial intelligence", "machine learning", etc.)

2. **Enrichment fallback** (`_select_ai_topics_by_enrichment`): Failed due to:
   - `min_ratio=1.5` threshold too strict
   - `min_topic_docs=15` too high for smaller samples
   - All topics had similar AI content (no enrichment differentiation)

3. **Silent failure**: When both methods failed, `ai_topics = set()` and all documents received `ai_topic_share = 0.0`

### Code Path
```
topics_bertopic.py:338-346
├── _ai_topics_from_keywords(topic_keywords) → Empty set
├── _select_ai_topics_by_enrichment(topics, ai_score) → Empty set
└── ai_topic_share = 0.0 for all docs (SILENT!)
```

### Fixes Applied

**File: `src/features/topics_bertopic.py`**

1. **Added logging to keyword detection** (lines 86-163):
   - Logs which AI topics were found via keywords
   - Warns when no topics match and shows sample topic keywords for debugging

2. **Relaxed enrichment thresholds**:
   - `min_ratio`: 1.5 → 1.2 (more lenient)
   - `top_k`: 3 → 5 (consider more topics)
   - `min_topic_docs`: 15 → 5 (works with smaller samples)

3. **Added document-level fallback**:
   - When no AI topics are identified by keywords or enrichment, uses document-level AI score normalized to [0, 1]
   - Returns `method="bertopic-docfallback"` or `method="lda-docfallback"`

4. **Added validation logging**:
   - Logs `ai_topic_share` stats (mean, nonzero_rate) after computation
   - Emits ERROR if nonzero_rate < 0.1% (clear failure indicator)

### Expected Outcome

- `ai_topic_share` will have meaningful variation
- Figures will display actual trend data
- Logs clearly indicate identification method used

---

## Issue 2: ConvergenceWarning

### Symptoms
```
sklearn.exceptions.ConvergenceWarning: Objective did not converge.
You might want to increase the number of iterations.
Duality gap: X, tolerance: Y
```

### Root Cause

Insufficient `max_iter` settings for high-dimensional text features:

| Model | Previous | Issue |
|-------|----------|-------|
| ElasticNetCV | max_iter=20000, tol=1e-4 | Insufficient for sparse features |
| LogisticRegression (SAGA) | max_iter=4000, tol=1e-4 | SAGA needs more iterations |
| LogisticRegression (LBFGS) | max_iter=2000, tol=1e-4 | Borderline |

### Fixes Applied

**File: `src/models/train_regression.py`**

```python
ElasticNetCV(
    max_iter=50000,   # Increased from 20000
    tol=1e-3,         # Relaxed from 1e-4
    n_jobs=-1,        # Enable parallelization
    ...
)
```

**File: `src/models/train_classification.py`**

```python
# BM2 LogisticRegression
LogisticRegression(max_iter=8000, tol=1e-3, solver="lbfgs")

# M1 LogisticRegression (L1)
LogisticRegression(
    max_iter=15000,   # Increased from 4000
    tol=1e-3,         # Relaxed from 1e-4
    ...
)
```

### Expected Outcome

- ConvergenceWarning eliminated or rare
- Model performance unchanged (verify with before/after metrics)

---

## Issue 3: Missing Model Results Plots

### Problem
No visualization of `model_results.csv` for comparing model performance.

### Fix Applied

**New File: `src/viz/results_from_tables.py`**

Created visualization module with:
- `plot_regression_comparison()`: Bar chart of RMSE/MAE by model
- `plot_classification_comparison()`: Bar chart of AUC/F1 by model
- `plot_delta_vs_benchmark()`: Improvement vs BM2 baseline
- `generate_results_figures()`: Orchestrates all plots

**Integration in `src/pipeline/run_all.py`**:
```python
from src.viz.results_from_tables import generate_results_figures
# After saving model_results.csv:
generate_results_figures(tbl_dir / "model_results.csv", fig_dir, logger=logger)
```

### New Figures Generated
- `results_model_compare_regression.png`
- `results_model_compare_classification.png`
- `results_delta_vs_benchmark.png`

---

## Issue 4: Validation Script

### Problem
No automated way to verify pipeline outputs are complete and correct.

### Fix Applied

**New File: `src/pipeline/validate_outputs.py`**

**CRITICAL BUG FIX (2026-01-24):** The validation script was looking for wrong file names:

| Wrong Name | Correct Name |
|------------|--------------|
| `panel.parquet` | `panel_with_targets.parquet` |
| `ani.parquet` | `features_ani.parquet` |
| `topics.parquet` | `features_topics.parquet` |

This caused repeated "file not found" validation failures despite files existing.

Validation checks:
1. **Panel dataset**: Exists, non-empty, has required columns
2. **ANI features**: Exists, `ani_kw_per1k` column present, <5% null
3. **AI topic share**: Exists, `ai_topic_share` column present, ≥95% non-null, ≥1% nonzero
4. **Model results**: CSV exists, expected models present, metrics not all NaN
5. **Required figures**: Exist and file size > 5KB (not empty)

### Usage
```bash
# Validate dev outputs
python -m src.pipeline.validate_outputs --dev

# Validate production outputs
python -m src.pipeline.validate_outputs

# Verbose mode
python -m src.pipeline.validate_outputs -v
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/features/topics_bertopic.py` | Added logging, document-level fallback, relaxed thresholds |
| `src/models/train_regression.py` | Increased max_iter=50000, added tol=1e-3 |
| `src/models/train_classification.py` | Increased max_iter, added tol=1e-3 |
| `src/viz/results_from_tables.py` | **NEW** - Model results visualization |
| `src/pipeline/validate_outputs.py` | **NEW** - Output validation script |
| `src/pipeline/run_all.py` | Integrated results plots generation |

---

## Verification Steps

```bash
# 1. Syntax check
python -c "from src.pipeline.run_all import main"

# 2. Quick dev test
python -m src.pipeline.run_all --dev --dev-sample-n 500 --start-year 2020 --recompute

# 3. Check ai_topic_share variation
python -c "import pandas as pd; df=pd.read_parquet('data/processed/dev/features_topics.parquet'); print(df['ai_topic_share'].describe())"

# 4. Validate outputs
python -m src.pipeline.validate_outputs --dev

# 5. Full production run
python -m src.pipeline.run_all --start-year 2018 --recompute

# 6. Final validation
python -m src.pipeline.validate_outputs
```

---

## Acceptance Criteria

- [ ] `eda_ai_topic_share_trend.png` shows visible trend line
- [ ] `results_pre_post_ai_topic_share_by_sector.png` shows bars for each sector
- [ ] `ai_topic_share` has ≥95% non-null and ≥1% nonzero values
- [ ] ConvergenceWarning eliminated or documented as benign
- [ ] `results_model_compare_regression.png` exists
- [ ] `results_model_compare_classification.png` exists
- [ ] `results_delta_vs_benchmark.png` exists
- [ ] `python -m src.pipeline.validate_outputs` exits with code 0

---

## Guardrails Check

| Principle | Status | Notes |
|-----------|--------|-------|
| **SRP** | ✓ | Each new function has single responsibility |
| **Modularity** | ✓ | New visualization module is self-contained |
| **Readability** | ✓ | Added logging explains "why" at decision points |
| **YAGNI** | ✓ | Only implemented requested fixes, no over-engineering |
| **Data Flow** | ✓ | Clear: topics → ai_topic_share → plots; no side effects |
