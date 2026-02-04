# AI Narratives in S&P 500 Earnings Calls

End-to-end, reproducible research pipeline for the proposal **“AI Narratives in S&P 500 Earnings Calls”**.

## Quickstart

Create an environment (Python 3.10+ recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

Optional (preferred) BERTopic topic modeling:

```bash
pip install -r requirements-bertopic.txt
```

Run the full pipeline:

```bash
python -m src.pipeline.run_all
```

This now includes a mandatory transfer-learning stage:
- Source domain: Kaggle `jannalipenkova/ai-media-dataset` (`tags`-based weak supervision)
- Target domain: earnings call transcripts (`transfer_ai_prob`, `transfer_ai_logit`, `transfer_ai_confidence`)

GPU acceleration (recommended on Windows + NVIDIA CUDA):

- Install a CUDA-enabled PyTorch build (follow the official PyTorch installer for your CUDA version).
- Run with CUDA (default):

```bash
python -m src.pipeline.run_all --device cuda
```

If you do not have CUDA available, use CPU:

```bash
python -m src.pipeline.run_all --device cpu
```

Run a fast DEV sample (for iteration):

```bash
python -m src.pipeline.run_all --dev --dev-sample-n 800
```

Quick smoke test (synthetic data, no HF download required):

```bash
python -m src.pipeline.smoke_test --rows 400
```

Recompute everything (ignore caches):

```bash
python -m src.pipeline.run_all --recompute
```

Force retraining of the transfer encoder:

```bash
python -m src.pipeline.run_all --transfer-retrain
```

Tune transfer training/runtime:

```bash
python -m src.pipeline.run_all --transfer-device auto --transfer-epochs 2 --transfer-batch-size 16 --transfer-lr 2e-5
```

## Research Design (Implemented)

**Dataset**
- Hugging Face Hub: `glopardo/sp500-earnings-transcripts` via `datasets.load_dataset`.
- Unit: firm-quarter (`ticker`, `datacqtr`).

**Targets**
- Regression (main): `delta_peforw_qavg = peforw_qavg(t+1) - peforw_qavg(t)`
- Regression (robustness): `delta_peforw_eoq = peforw_eoq(t+1) - peforw_eoq(t)`
- Classification: `valuation_upgrade = 1` if `delta_peforw_qavg` is in the top quartile within (`sector`, `datacqtr`)

Note: for regression model fitting/evaluation we winsorize the regression target at the 1st/99th percentiles of the **training** split to reduce extreme P/E outliers (common when earnings are near zero or negative). The raw target columns are still preserved in the dataset.

**Text Features**
- AI dictionary intensity (ANI): keyword counts per 1k tokens (plus subcomponents)
- Topic-based AI share: BERTopic if available, else gensim LDA
- Semantic similarity: sentence-transformers embeddings vs AI seed statements (TF-IDF fallback if unavailable)
- Sentiment/uncertainty: Loughran–McDonald Positive/Negative/Uncertainty per 1k tokens
- Transfer-learning encoder features from AI media source task:
  - `transfer_ai_prob`
  - `transfer_ai_logit`
  - `transfer_ai_confidence`

**Controls**
- Length controls: `n_tokens`, `n_chars`
- Lagged valuation/financials: `lag_peforw_*`, `lag_eps*`
- Fixed effects proxies: `sector`, `datacqtr` dummies; `ticker` dummies in linear models

**No leakage evaluation**
- Primary time split: train `<= 2022Q3`, test `>= 2022Q4`.
  - Motivation: `2022Q4` aligns with the broad diffusion of generative AI narratives (ChatGPT era), making it a natural out-of-sample boundary.
- Rolling/forward validation is also produced for robustness.

## Outputs

Running `python -m src.pipeline.run_all` produces:
- `data/processed/panel_with_targets.parquet` (or `data/processed/dev/panel_with_targets.parquet` in DEV mode)
- `data/processed/modeling_dataset.parquet` with merged features
- `data/processed/features_transfer.parquet` transfer-learning features
- `outputs/figures/` publication-ready figures (EDA + results)
- `outputs/tables/model_results.csv` benchmark comparison (BM1, BM2, M1, M2)
- `outputs/tables/model_results_transfer_ablation.csv` baseline vs `+transfer` ablation table
- `outputs/tables/transfer/source_metrics.csv` source-task metrics (AUC/F1/PR-AUC/Accuracy)
- `outputs/tables/rolling_results.csv` rolling validation (BM2 vs M1)
- `outputs/models/` saved artifacts (topic model, TF-IDF, fitted sklearn models)
- `outputs/models/transfer/encoder/` transfer encoder + tokenizer artifacts
- `outputs/logs/pipeline.log`

## Notes

**Loughran–McDonald dictionary**
- The pipeline downloads the latest master dictionary CSV from Notre Dame’s public “Software Repository for Accounting and Finance” page (Google Drive link) and caches it under `data/external/lm_dict/`.

**Topic modeling**
- Default mode is `auto`: use BERTopic if installed; otherwise LDA.

**Caching**
- Hugging Face caching uses `data/hf_cache/`.
- Most intermediate outputs are cached as parquet under `data/processed/` and recomputed only with `--recompute`.
- Document embeddings are computed once and cached under `data/processed/**/document_embeddings.npy`, then reused for both BERTopic and semantic similarity.
- AI media source data and labels are cached at:
  - `data/external/transfer/ai_media_raw.parquet`
  - `data/external/transfer/ai_media_labeled.parquet`
- Transfer encoder artifacts are cached under `outputs/models/transfer/encoder/` and reused unless `--transfer-retrain` is set.

**Performance knobs**
- `--emb-batch-size`: increase on GPUs if you have VRAM (e.g., 64 → 128).
- `--emb-max-chunks`: cap transcript chunks for faster embeddings (0 = no cap; lower is faster).
- `--no-topic-probs`: faster BERTopic (but `ai_topic_share` becomes a 0/1 indicator).

## Troubleshooting

If `load_dataset("glopardo/sp500-earnings-transcripts")` fails due to network restrictions (some networks block Hugging Face large-file backends), you can manually download the parquet shards from the dataset repo and set:

```bash
export SP500_TRANSCRIPTS_LOCAL_DIR="/path/to/folder/with/parquet/files"
```

Then re-run `python -m src.pipeline.run_all`.

If Kaggle download fails, verify Kaggle credentials are configured for `kagglehub` and retry.
