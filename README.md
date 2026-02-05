# AI Narratives in S&P 500 Earnings Calls

End-to-end, reproducible research pipeline for **descriptive Q&A trend analysis** in earnings calls.

## Quickstart

Create an environment (Python 3.10+ recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

Optional (preferred) BERTopic topic modeling:

```bash
pip install -r requirements-bertopic.txt
```

Run the full QA pipeline:

```bash
python -m src.pipeline.run_all_qa
```

Optional transfer-learning AI encoder:
- Source domain: Kaggle `jannalipenkova/ai-media-dataset` (`tags`-based weak supervision)
- Target domain: earnings call transcripts (used as `ai_score_encoder` if present)

Run a fast DEV sample (for iteration):

```bash
python -m src.pipeline.run_all_qa --dev --dev-sample-n 800
```

Recompute everything (ignore caches):

```bash
python -m src.pipeline.run_all_qa --recompute
```

## Research Design (Implemented)

**Dataset**
- Hugging Face Hub: `glopardo/sp500-earnings-transcripts` via `datasets.load_dataset`.
- Unit: call (`ticker`, `datacqtr`, `earnings_date`).

**Objective**
- No forecasting. Focus on descriptive Q&A trends:
  - AI question/answer rates by quarter
  - Who introduces AI first (analyst vs management)
  - Uncertainty differences in AI vs non-AI answers

**Text Features**
- AI keyword heuristic (high precision)
- Transfer-learning encoder score (higher recall) if available
- Loughran–McDonald uncertainty for answer sentiment

**Outputs**
- `data/processed/calls.parquet`
- `data/processed/turns.parquet`
- `data/processed/qa_pairs.parquet`
- `outputs/tables/*.csv`
- `outputs/figures/*.png`

**Legacy models**
- Forecasting code is quarantined under `src/legacy/` and not used in the QA pipeline.

## Troubleshooting

If `load_dataset("glopardo/sp500-earnings-transcripts")` fails due to network restrictions, you can manually download the parquet shards from the dataset repo and set:

```bash
export SP500_TRANSCRIPTS_LOCAL_DIR="/path/to/folder/with/parquet/files"
```

Then re-run `python -m src.pipeline.run_all_qa`.
