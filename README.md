# S&P 500 AI Narrative Text Mining

Analysis of AI narratives in S&P 500 earnings call transcripts using dictionary methods, topic modeling, and conversational dynamics (2020-2025).

## 🎯 Project Overview

This project uses **text mining** techniques to analyze how S&P 500 companies discuss AI in their quarterly earnings calls. We apply:

1. **Dictionary Detection**: Curated keyword/regex matching for AI-related content
2. **Topic Modeling**: Quarterly LDA topics for AI-focused text (manual naming supported)
3. **Conversational Dynamics**: Analyze who initiates AI discussions (management vs analysts)
4. **Quadrant Analysis**: Classify companies by their AI narrative patterns

## 📁 Project Structure

```
TEXT-MINING/
├── src/
│   ├── preprocessing/          # Text preprocessing
│   │   ├── transcript_parser.py    # Speech/Q&A splitting
│   │   └── sentence_splitter.py    # Sentence tokenization
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

# Compute metrics
python -m src.metrics.ai_intensity --input outputs/features/sentences_with_keywords.parquet

# Topic modeling (quarterly)
python -m src.analysis.topic_modeling --sentences outputs/features/sentences_with_keywords.parquet --filter-ai
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
  --wrds            WRDS financial metadata
  --dev             Development mode
  --ai-method       kw | topic
  --kw-workers      Keyword detection workers
  --metrics-workers AI intensity workers
```

## 📚 Data Sources

1. **S&P 500 Earnings Transcripts** - Hugging Face: `kurry/sp500_earnings_transcripts`
2. **WRDS Compustat** - `Sp500_meta_data.csv`

## 🧪 Research Questions

1. **RQ1**: How has AI discussion intensity changed over time (esp. post-ChatGPT)?
2. **RQ2**: Who initiates AI discussions - management or analysts?
3. **RQ3**: What firm characteristics predict management proactiveness on AI?

## 📄 License

Academic research project for Text Mining for Economics and Finance.
