# S&P 500 AI Narrative Text Mining

Analysis of AI narratives in S&P 500 earnings call transcripts using transfer learning and conversational dynamics (2020-2025).

## 🎯 Project Overview

This project uses **text mining** techniques to analyze how S&P 500 companies discuss AI in their quarterly earnings calls. We apply:

1. **Transfer Learning**: Fine-tune FinBERT on AI news articles, then apply to earnings transcripts
2. **Conversational Dynamics**: Analyze who initiates AI discussions (management vs analysts)
3. **Quadrant Analysis**: Classify companies by their AI narrative patterns

## 📁 Project Structure

```
TEXT-MINING/
├── src/
│   ├── preprocessing/          # Text preprocessing
│   │   ├── transcript_parser.py    # Speech/Q&A splitting
│   │   └── sentence_splitter.py    # Sentence tokenization
│   ├── models/                 # ML models
│   │   ├── ai_news_dataset.py      # Training data loader
│   │   ├── ai_classifier.py        # FinBERT classifier
│   │   └── predict.py              # Batch inference
│   ├── metrics/                # Text mining metrics
│   │   ├── ai_intensity.py         # AI intensity scores
│   │   └── initiation_score.py     # Who starts AI discussions
│   ├── baselines/              # Baseline methods
│   │   └── keyword_detector.py     # Dictionary-based detection
│   └── analysis/               # Statistical analysis
│       ├── time_series.py          # Trend analysis
│       ├── company_quadrants.py    # Company classification
│       └── regression.py           # Cross-sectional regression
├── tests/                      # Unit tests
├── outputs/
│   ├── models/                 # Saved model checkpoints
│   ├── features/               # Computed metrics
│   └── figures/                # Plots and tables
├── run_pipeline.py             # Main orchestration script
├── requirements.txt            # Dependencies
└── Proposal.md                 # Research proposal
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Production run (full dataset)
python run_pipeline.py

# Development mode (small sample)
python run_pipeline.py --dev --dev-sample 100
```

### 3. Run Individual Steps

```bash
# Parse transcripts
python -m src.preprocessing.transcript_parser --input final_dataset.parquet

# Train classifier
python -m src.models.ai_classifier --train-data outputs/features/ai_news_train.parquet

# Run inference
python -m src.models.predict --sentences outputs/features/sentences.parquet

# Compute metrics
python -m src.metrics.ai_intensity --input outputs/features/sentences_with_predictions.parquet
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## 📊 Key Metrics

| Metric | Description |
|--------|-------------|
| **Speech AI Intensity** | % of sentences in prepared remarks mentioning AI |
| **Q&A AI Intensity** | % of sentences in Q&A section mentioning AI |
| **AI Initiation Score** | Management-driven (1.0) vs analyst-driven (0.0) |

## 📈 Company Quadrants

| Quadrant | Speech | Q&A | Interpretation |
|----------|--------|-----|----------------|
| Aligned | High | High | Genuine AI focus |
| Passive | Low | High | Responding to analyst pressure |
| Self-Promoting | High | Low | AI-washing? |
| Silent | Low | Low | Not engaging with AI |

## 🔧 Key Parameters

```bash
python run_pipeline.py --help

Options:
  --input           Input dataset (default: final_dataset.parquet)
  --ai-news         AI news training data
  --wrds            WRDS financial metadata
  --epochs          Training epochs (default: 3)
  --batch-size      Batch size (default: 16)
  --dev             Development mode
  --skip-training   Use existing model
```

## 📚 Data Sources

1. **S&P 500 Earnings Transcripts** - Hugging Face: `kurry/sp500_earnings_transcripts`
2. **AI News Dataset** - `ai_media_dataset_20250911.csv` 
3. **WRDS Compustat** - `Sp500_meta_data.csv`

## 🧪 Research Questions

1. **RQ1**: How has AI discussion intensity changed over time (esp. post-ChatGPT)?
2. **RQ2**: Who initiates AI discussions - management or analysts?
3. **RQ3**: What firm characteristics predict management proactiveness on AI?

## 📄 License

Academic research project for Text Mining for Economics and Finance.
