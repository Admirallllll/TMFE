# Transfer Learning Methodology

## Overview
This project uses a two-stage transfer-learning setup:

1. **Source task (AI Media domain)**  
Train an encoder on Kaggle `jannalipenkova/ai-media-dataset` using `tags`-driven labels.

2. **Target task (Earnings Call domain)**  
Run the trained encoder on earnings-call transcripts and add transfer features into downstream valuation models.

## Data Flow

```text
Kaggle AI Media dataset
  -> normalize tags
  -> high-precision weak labeling (positive / negative / review)
  -> train transfer encoder (binary classifier)
  -> save encoder + threshold + source metrics

Earnings call transcripts
  -> preprocess text
  -> transfer encoder inference
  -> transfer_ai_prob / transfer_ai_logit / transfer_ai_confidence
  -> merged with ANI + sentiment + topic + similarity features
  -> regression/classification benchmarks and ablation
```

## Labeling Policy
- **High precision first**: ambiguous or conflicting tag combinations are excluded from source-task training.
- Strong positive and strong negative tags have larger weights than weak tags.
- Conflict cases (`POS_STRONG` and `NEG_STRONG` both hit) go to review pool (`label=None`).

## Output Artifacts
- `data/external/transfer/ai_media_raw.parquet`
- `data/external/transfer/ai_media_labeled.parquet`
- `outputs/tables/transfer/tag_audit.csv`
- `outputs/models/transfer/encoder/*`
- `outputs/tables/transfer/source_metrics.csv`
- `data/processed/features_transfer.parquet`
- `outputs/tables/model_results_transfer_ablation.csv`

## Suggested Report Structure
1. **Motivation**
   Why transfer learning is needed for AI narrative detection in earnings calls.
2. **Source Task**
   Dataset, tag-based labeling rule, class balance, and source metrics.
3. **Target Task**
   How transfer features are injected into baseline models.
4. **Ablation**
   Compare `M1/M2` vs `M1+transfer/M2+transfer`.
5. **Error Analysis**
   False positive/false negative cases from transfer inference.
6. **Limitations**
   Weak-label noise, domain mismatch, and future improvements.
