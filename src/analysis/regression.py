"""
Regression Analysis Module

Cross-sectional regression analysis:
- DV: AI Initiation Score (management proactiveness)
- IVs: R&D Intensity, Lagged Returns, Beat/Miss Earnings, Market Cap, Industry
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os


def prepare_regression_data(
    initiation_scores_path: str,
    doc_metrics_path: str,
    wrds_data_path: str
) -> pd.DataFrame:
    """
    Prepare data for regression analysis by merging metrics with financial data.
    
    Args:
        initiation_scores_path: Path to initiation scores
        doc_metrics_path: Path to document metrics
        wrds_data_path: Path to WRDS metadata
        
    Returns:
        Merged DataFrame ready for regression
    """
    print("Loading data...")
    
    # Load data
    initiation = pd.read_parquet(initiation_scores_path)
    doc_metrics = pd.read_parquet(doc_metrics_path)
    wrds = pd.read_csv(wrds_data_path, low_memory=False)
    
    # Parse doc_id into ticker, year, quarter
    def parse_doc_id(doc_id):
        parts = str(doc_id).rsplit('_', 1)
        if len(parts) == 2:
            ticker = parts[0]
            yq = parts[1]
            if 'Q' in yq:
                year = int(yq.split('Q')[0])
                quarter = int(yq.split('Q')[1])
                return ticker, year, quarter
        return None, None, None
    
    # Add parsed columns
    initiation['_parsed'] = initiation['doc_id'].apply(parse_doc_id)
    initiation['ticker'] = [p[0] for p in initiation['_parsed']]
    initiation['year'] = [p[1] for p in initiation['_parsed']]
    initiation['quarter'] = [p[2] for p in initiation['_parsed']]
    initiation = initiation.drop('_parsed', axis=1)
    
    # Same for doc_metrics
    doc_metrics['_parsed'] = doc_metrics['doc_id'].apply(parse_doc_id)
    doc_metrics['ticker'] = [p[0] for p in doc_metrics['_parsed']]
    doc_metrics['year'] = [p[1] for p in doc_metrics['_parsed']]
    doc_metrics['quarter'] = [p[2] for p in doc_metrics['_parsed']]
    doc_metrics = doc_metrics.drop('_parsed', axis=1)
    
    # Merge initiation and doc_metrics
    merged = pd.merge(
        initiation,
        doc_metrics[['doc_id', 'speech_kw_ai_ratio', 'qa_kw_ai_ratio', 'overall_kw_ai_ratio']],
        on='doc_id',
        how='left'
    )
    
    # Prepare WRDS data (need to match on ticker + quarter)
    wrds = wrds.rename(columns={'tic': 'ticker'})
    if 'datadate' in wrds.columns:
        wrds['datadate'] = pd.to_datetime(wrds['datadate'], errors='coerce')
    
    # Parse WRDS quarter
    if 'datacqtr' in wrds.columns:
        wrds['wrds_year'] = wrds['datacqtr'].str[:4].astype(int)
        wrds['wrds_quarter'] = wrds['datacqtr'].str[-1].astype(int)
    
    # Create financial metrics
    # R&D Intensity = R&D / Market Value
    wrds['rd_intensity'] = wrds['xrdq'] / wrds['mkvaltq']
    wrds['rd_intensity'] = wrds['rd_intensity'].replace([np.inf, -np.inf], np.nan)
    
    # Log Market Cap
    wrds['log_mktcap'] = np.log(wrds['mkvaltq'].replace(0, np.nan))
    
    # Stock Return (simplistic: price change, would need lag data for proper calc)
    # For now, just use price as proxy
    wrds['stock_price'] = wrds['prccq']
    
    # EPS Beat/Miss (simplified: positive EPS = beat)
    wrds['eps_positive'] = (wrds['epspxq'] > 0).astype(int)
    
    # Industry dummies (using GICS sector)
    if 'gsector' in wrds.columns:
        wrds['sector'] = wrds['gsector'].astype(str)
    
    # Select WRDS columns for merge
    wrds_cols = ['ticker', 'wrds_year', 'wrds_quarter',
                 'rd_intensity', 'log_mktcap', 'stock_price', 
                 'eps_positive', 'sector', 'mkvaltq', 'xrdq', 'epspxq']
    wrds_subset = wrds[wrds_cols + (['datadate'] if 'datadate' in wrds.columns else [])].copy()
    # Ensure one row per (ticker, year, quarter) to avoid duplicate-merge explosions
    if 'datadate' in wrds_subset.columns:
        wrds_subset = wrds_subset.sort_values(['datadate'])
        wrds_subset = wrds_subset.drop_duplicates(subset=['ticker', 'wrds_year', 'wrds_quarter'], keep='last')
        wrds_subset = wrds_subset.drop(columns=['datadate'])
    else:
        wrds_subset = wrds_subset.drop_duplicates(subset=['ticker', 'wrds_year', 'wrds_quarter'], keep='last')
    
    # Merge with main data
    final = pd.merge(
        merged,
        wrds_subset,
        left_on=['ticker', 'year', 'quarter'],
        right_on=['ticker', 'wrds_year', 'wrds_quarter'],
        how='left'
    )
    
    print(f"Final regression dataset: {len(final)} observations")
    print(f"Missing rd_intensity: {final['rd_intensity'].isna().sum()}")
    print(f"Missing log_mktcap: {final['log_mktcap'].isna().sum()}")
    
    return final


def run_regression(
    df: pd.DataFrame,
    dv: str,
    ivs: List[str],
    add_constant: bool = True,
    robust: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run OLS regression.
    
    Args:
        df: DataFrame with variables
        dv: Dependent variable column name
        ivs: List of independent variable column names
        add_constant: Whether to add intercept
        robust: Whether to use robust standard errors
        
    Returns:
        statsmodels regression results
    """
    # Drop missing values
    cols = [dv] + ivs
    reg_df = df[cols].dropna()
    
    print(f"Regression sample size: {len(reg_df)}")
    
    y = reg_df[dv]
    X = reg_df[ivs]
    
    if add_constant:
        X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    
    if robust:
        results = model.fit(cov_type='HC1')  # Heteroskedasticity-robust
    else:
        results = model.fit()
    
    return results


def run_regression_analysis(
    initiation_scores_path: str,
    doc_metrics_path: str,
    wrds_data_path: str,
    output_dir: str = "outputs/figures"
) -> Dict:
    """
    Full regression analysis pipeline.
    
    Args:
        initiation_scores_path: Path to initiation scores
        doc_metrics_path: Path to document metrics
        wrds_data_path: Path to WRDS data
        output_dir: Output directory
        
    Returns:
        Dictionary with regression results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    reg_df = prepare_regression_data(
        initiation_scores_path,
        doc_metrics_path,
        wrds_data_path
    )
    
    # Save regression dataset
    reg_df.to_parquet(f"{output_dir}/../features/regression_dataset.parquet", index=False)
    
    results = {}
    
    # Model 1: Basic - AI Initiation Score ~ Size + R&D
    print("\n" + "="*60)
    print("Model 1: AI Initiation Score ~ Size + R&D Intensity")
    print("="*60)
    
    model1 = run_regression(
        reg_df,
        dv='ai_initiation_score',
        ivs=['log_mktcap', 'rd_intensity']
    )
    print(model1.summary())
    results['model1'] = model1
    
    # Model 2: Add EPS
    print("\n" + "="*60)
    print("Model 2: Add EPS (Beat/Miss)")
    print("="*60)
    
    model2 = run_regression(
        reg_df,
        dv='ai_initiation_score',
        ivs=['log_mktcap', 'rd_intensity', 'eps_positive']
    )
    print(model2.summary())
    results['model2'] = model2
    
    # Model 3: AI Intensity as DV (alternative)
    print("\n" + "="*60)
    print("Model 3: Overall AI Ratio (KW) ~ Financial Metrics")
    print("="*60)
    
    model3 = run_regression(
        reg_df,
        dv='overall_kw_ai_ratio',
        ivs=['log_mktcap', 'rd_intensity', 'eps_positive']
    )
    print(model3.summary())
    results['model3'] = model3
    
    # Create summary table
    summary = summary_col(
        [model1, model2, model3],
        stars=True,
        float_format='%0.4f',
        model_names=['AI Initiation (1)', 'AI Initiation (2)', 'AI Ratio (3, KW)']
    )
    
    with open(f"{output_dir}/regression_summary.txt", 'w') as f:
        f.write(str(summary))
    print(f"\nSaved regression summary to {output_dir}/regression_summary.txt")
    
    # Plot coefficient comparison
    plot_coefficients(results, f"{output_dir}/regression_coefficients.png")
    
    return results


def plot_coefficients(
    results: Dict,
    output_path: str
):
    """
    Plot regression coefficients with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract coefficients from model2 (most complete)
    model = results['model2']
    params = model.params.drop('const')
    conf_int = model.conf_int().drop('const')
    
    y_pos = range(len(params))
    
    ax.barh(y_pos, params.values, xerr=[
        params.values - conf_int[0].values,
        conf_int[1].values - params.values
    ], capsize=5, color='steelblue', alpha=0.7)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params.index)
    ax.set_xlabel('Coefficient Estimate')
    ax.set_title('Regression Coefficients: AI Initiation Score')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved coefficient plot to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Regression analysis")
    parser.add_argument("--initiation", default="outputs/features/initiation_scores.parquet")
    parser.add_argument("--doc-metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--wrds", default="Sp500_meta_data.csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    
    args = parser.parse_args()
    
    run_regression_analysis(
        args.initiation,
        args.doc_metrics,
        args.wrds,
        args.output_dir
    )
