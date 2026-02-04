# Proposal Supplement: Data and Methodology Details

## 1. Data Source

**Primary Dataset**: S&P 500 Earnings Call Transcripts
**Source**: Hugging Face Datasets
**URL**: https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts

The dataset is publicly available and can be accessed directly through the Hugging Face platform.

## 2. Data Acquisition Plan

The data is obtained programmatically using the Hugging Face `datasets` library:

```python
from datasets import load_dataset
dataset = load_dataset("glopardo/sp500-earnings-transcripts")
```

No manual download or web scraping is required. The dataset is well-structured and includes both transcript text and associated metadata (ticker, sector, industry, financial metrics, etc.).

## 3. Current Progress

We have downloaded the dataset from Hugging Face and loaded it into our Python environment for initial exploration. We are currently examining the data structure, checking for missing values, and understanding the distribution of key variables across sectors and time periods.

## 4. Dataset Statistics

| Metric | Value |
|--------|-------|
| **Number of documents** | 12,774 firm-quarter transcripts |
| **Time span** | 2018 Q1 – 2024 Q4 (28 quarters) |
| **Unique firms** | 494 S&P 500 companies |
| **Number of sectors** | 11 |
| **Average document length** | ~9,177 tokens (~52,800 characters) |
| **Document length range** | 769 – 43,339 tokens |

## 5. Population and Outcomes of Interest

### Population

S&P 500 constituent firms, representing large-cap U.S. equities. This sample provides:
- High data quality (mandatory SEC filings, consistent reporting standards)
- Broad sector coverage across the U.S. economy
- Sufficient variation in AI adoption and discussion intensity

### Outcomes of Interest

| Target | Type | Definition |
|--------|------|------------|
| **Target A** | Regression | Next-quarter change in forward P/E ratio: ΔForward P/E = peforw_qavg(t+1) − peforw_qavg(t) |
| **Target B** | Classification | Valuation upgrade indicator: 1 if ΔForward P/E is in the top 25% within the same sector and quarter; else 0 |

## 6. Control Variables

| Category | Variables | Description |
|----------|-----------|-------------|
| **Document controls** | Token count, character count | All intensity measures are normalised by document length to ensure comparability |
| **Lagged fundamentals** | Prior-quarter forward P/E, trailing EPS | Controls for baseline valuation levels and earnings trajectory |
| **Fixed effects** | Sector dummies, year-quarter dummies | Absorbs sector-specific and time-specific unobserved heterogeneity |
| **Industry metadata** | Sector, industry classification | Allows for sector-level heterogeneity analysis |

### Additional Context Variables

- **Forward P/E variants**: `peforw_qavg` (quarterly average), `peforw_eoq` (end-of-quarter)
- **Forward EPS variants**: `eps12mfwd_qavg` (12-month forward EPS, quarterly average)
- **Time indicators**: `earnings_date`, `year`, `quarter`, `datacqtr`
- **Firm identifiers**: `ticker`, `company`, `cik`
