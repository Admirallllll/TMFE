# S&P 500 AI Narrative Text Mining

Analysis of AI narratives in S&P 500 earnings call transcripts using dictionary methods, topic modeling, and advanced econometric mechanism analysis (2020-2025).

## 🎯 Project Overview

This project uses **text mining** techniques to analyze how S&P 500 companies discuss AI in their quarterly earnings calls. Through continuous development, our pipeline addresses **AI-Washing phenomena**, outperformance classification, and dual-mechanism channel analysis.

1. **Dictionary Detection & Parsing**: Curated keyword/regex matching for AI-related content and speaker separation.
2. **Conversational Dynamics**: Analyze who initiates AI discussions (management vs analysts) and map companies into behavioral quadrants.
3. **Sparse Text Modeling (Lasso)**: N-gram extraction combined with positive/negative sentiment ratios to predict forward-looking R&D intensity.
4. **Outperformance Benchmark**: Classification benchmarks comparing metadata vs text ratios for predicting whether a firm beats the sector median.
5. **Dual-Mechanism Path Analysis**: Econometric models testing if AI discussions translate to short-term profitability (EPS Growth via Efficiency-AI) vs long-term innovation (R&D Increase via Growth-AI).

## 📁 Project Structure

```
TEXT-MINING/
├── src/
│   ├── preprocessing/          # Data download, Transcript parsing, Sentence splitting
│   ├── metrics/                # Text mining metrics (AI intensity, Initiation score)
│   ├── baselines/              # Dictionary-based keyword detection
│   └── analysis/               # Core advanced analysis modules
│       ├── time_series.py             # Trend analysis
│       ├── company_quadrants.py       # Company behavioral classification
│       ├── regression.py              # Cross-sectional basic regression
│       ├── benchmark_comparison.py    # [Stage 11] Classification: Metadata vs Text Ratio
│       ├── lasso_text_features.py     # [Stage 12] AI n-grams & R&D predictions (Volcano plots)
│       └── research_report.py         # [Stage 14] Dual-Path mechanisms (Efficiency vs Growth)
├── scripts/                    # Helper scripts (Extremes, manual validation)
├── tests/                      # Unit tests
├── outputs/
│   ├── features/               # Computed metrics (.parquet)
│   ├── figures/                # Plots, Volcano charts, ROC curves
│   └── report/                 # Auto-generated Research-Grade markdown reports
├── run_pipeline.py             # Main end-to-end orchestration script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Requirements & Configuration

A valid Hugging Face dataset access token is required to download and access the S&P 500 earnings call dataset.
Create a `.env` file in the root directory and add your token:

```bash
# Create and edit the .env file
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Production run (Full run from Stage 0 to Stage 14, fetches upstream data)
python run_pipeline.py --run-download

# Skip Stage 0 (Download) and rely on local `data/final_dataset.parquet`
python run_pipeline.py --start-stage 1

# Development mode (small sample: parses 100 docs per stage for rapid testing)
python run_pipeline.py --dev --dev-sample 100
```

### 3. Running Specific Pipeline Stages

The pipeline is divided into modular stages (0-14). You can resume or start execution from any specific stage using `--start-stage <int>`. 
For instance, if you modified the regression formula in Stage 14, you don't need to parse the text again:

```bash
# Rerun only the final Research Report (Stage 14) 
python run_pipeline.py --start-stage 14

# Start directly from Lasso Text Features (Stage 12)
python run_pipeline.py --start-stage 12

# Skip specific heavy visual models to speed up re-runs:
python run_pipeline.py --start-stage 10 --skip-lasso --skip-benchmark
```

### 4. Key Pipeline Arguments

```bash
Options:
  --input                   Path to final dataset (default: data/final_dataset.parquet)
  --wrds                    Path to WRDS mapping (default: data/wrds.csv)
  --run-download            Include Stage 0 (Fetch raw data from HuggingFace & Merge WRDS)
  --dev                     Run with small subsets for rapid testing
  --skip-lasso              Skip Stage 12 (Lasso Text Features / Volcano plots)
  --skip-benchmark          Skip Stage 11 (Outperformance classification)
  --skip-research-report    Skip Stage 14 (Dual-Path regressions & markdown report)
  --start-stage             Start from a specific stage (0-14)
```

## 🛠️ Advanced Tools & Scripts

The `scripts/` directory includes standalone analytical tools for deeper manual validation and data insight extractions. 

- **`manual_validation.py`**: Compares the pipeline's algorithmic labeling against double-adjudicated human ground-truth datasets. Useful to defend our algorithm's precision and kappa agreement scores.
  ```bash
  python scripts/manual_validation.py
  ```

- **`inspect_extremes.py` / `inspect_doc_extremes.py`**: Explores outliers and extreme data points among the transcripts. Need to find the company that talked about AI the absolute most? Or documents where AI sentences were purely analyst-driven? Run these scripts.
  ```bash
  # Check top documents with highest AI intensity metrics
  python scripts/inspect_doc_extremes.py
  ```

- **`export_annotation_samples.py`**: Samples random boundaries and chunks for building the ground-truth set for your human annotators. Used when retraining or refreshing the dictionary.

## 📊 Core Concepts & Findings

### 🤖 AI-Intensity & Quadrants
- **Speech vs Q&A AI Ratio:** Measures AI hype in prepared remarks vs analyst scrutiny.
- **Company Quadrants:** (Aligned, Passive, Self-Promoting, Silent) derived from the gap between Management push and Analyst pull.

### 📉 Mechanism Analysis (The "AI Washing" Check)
- **Efficiency-AI (Cost reduction/Automation):** Assessed against Next-Quarter EPS growth (YoY).
- **Growth-AI (Innovation/Revenue):** Assessed against Forward R&D Intensity Change.
- *Recent findings suggest significant AI-Washing, where extensive Growth-AI rhetoric fails to predict actual R&D budget expansions, producing null elasticities.*

### 🏆 Benchmark Modeling (Stage 11)
- Binary Outperformance: Does the firm beat the sector median Next-Quarter Market Cap Growth?
- Text/Ratio models are compared against pure Metadata baseline (Random Forest / Logistic L1) using strict Cross-Validation Folds and AUC-ROC evaluation.

## 📚 Data Sources

1. **S&P 500 Earnings Transcripts** - Extracted and structured text (2020-2025).
2. **WRDS Compustat** - Fundamental metrics (Log Market Cap, R&D Intensity, EPS).

## 📄 License

Academic research project for Text Mining for Economics and Finance.
