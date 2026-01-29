"""Validate pipeline outputs for completeness and correctness."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import Paths


def validate_panel(path: Path) -> tuple[bool, list[str]]:
    """Validate panel dataset."""
    errors = []
    if not path.exists():
        return False, [f"Panel file not found: {path}"]

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return False, [f"Failed to read panel: {e}"]

    if len(df) == 0:
        errors.append("Panel is empty (0 rows)")

    required_cols = ["ticker", "datacqtr", "quarter_index", "sector"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Panel missing columns: {missing}")

    return len(errors) == 0, errors


def validate_ai_topic_share(path: Path) -> tuple[bool, list[str]]:
    """Validate ai_topic_share feature."""
    errors = []
    if not path.exists():
        return False, [f"Topics file not found: {path}"]

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return False, [f"Failed to read topics: {e}"]

    if "ai_topic_share" not in df.columns:
        return False, ["ai_topic_share column not found"]

    col = df["ai_topic_share"]
    null_rate = col.isna().mean()
    zero_rate = (col == 0.0).mean()
    nonzero_rate = ((col > 0) & col.notna()).mean()

    if null_rate > 0.05:
        errors.append(f"ai_topic_share has {null_rate:.1%} null values (max 5%)")

    if nonzero_rate < 0.01:
        errors.append(f"ai_topic_share has only {nonzero_rate:.2%} nonzero values (min 1%)")

    if zero_rate > 0.99 and nonzero_rate < 0.01:
        errors.append("ai_topic_share is nearly all zeros - topic identification likely failed")

    return len(errors) == 0, errors


def validate_model_results(path: Path) -> tuple[bool, list[str]]:
    """Validate model_results.csv."""
    errors = []
    if not path.exists():
        return False, [f"Model results not found: {path}"]

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, [f"Failed to read model results: {e}"]

    expected_models = ["BM1_no_change", "BM2_metadata_ridge", "M1_text_elasticnet"]
    missing = [m for m in expected_models if m not in df["model"].values]
    if missing:
        errors.append(f"Missing expected models: {missing}")

    # Check regression models have RMSE
    reg = df.loc[df["task"] == "regression"]
    if not reg.empty and reg["rmse"].isna().all():
        errors.append("All regression RMSE values are NaN")

    # Check classification models have AUC
    cls = df.loc[df["task"] == "classification"]
    if not cls.empty and cls["auc"].isna().all():
        errors.append("All classification AUC values are NaN")

    return len(errors) == 0, errors


def validate_figures(figures_dir: Path) -> tuple[bool, list[str]]:
    """Validate required figures exist and have content."""
    errors = []
    required = [
        "eda_ai_topic_share_trend.png",
        "results_pre_post_ai_topic_share_by_sector.png",
    ]
    optional_but_expected = [
        "results_model_compare_regression.png",
        "results_model_compare_classification.png",
        "results_delta_vs_benchmark.png",
    ]

    for fig in required:
        path = figures_dir / fig
        if not path.exists():
            errors.append(f"Required figure not found: {fig}")
        elif path.stat().st_size < 5000:  # Less than 5KB likely means empty/error
            errors.append(f"Required figure suspiciously small ({path.stat().st_size} bytes): {fig}")

    for fig in optional_but_expected:
        path = figures_dir / fig
        if not path.exists():
            errors.append(f"Expected figure not found (optional): {fig}")

    return len(errors) == 0, errors


def validate_ani_features(path: Path) -> tuple[bool, list[str]]:
    """Validate ANI features."""
    errors = []
    if not path.exists():
        return False, [f"ANI file not found: {path}"]

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return False, [f"Failed to read ANI features: {e}"]

    if "ani_kw_per1k" not in df.columns:
        errors.append("ani_kw_per1k column not found")
        return False, errors

    col = df["ani_kw_per1k"]
    null_rate = col.isna().mean()
    if null_rate > 0.05:
        errors.append(f"ani_kw_per1k has {null_rate:.1%} null values (max 5%)")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline outputs")
    parser.add_argument("--dev", action="store_true", help="Check dev outputs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    paths = Paths()
    subdir = "dev" if args.dev else ""

    print(f"Validating {'DEV' if args.dev else 'PRODUCTION'} outputs...")
    print("=" * 60)

    all_passed = True
    # CRITICAL: Use correct file names that match run_all.py output
    checks = [
        ("Panel dataset", validate_panel, paths.processed_dir / subdir / "panel_with_targets.parquet"),
        ("ANI features", validate_ani_features, paths.processed_dir / subdir / "features_ani.parquet"),
        ("AI topic share", validate_ai_topic_share, paths.processed_dir / subdir / "features_topics.parquet"),
        ("Model results", validate_model_results, paths.outputs_dir / "tables" / subdir / "model_results.csv"),
        ("Required figures", validate_figures, paths.figures_dir / subdir),
    ]

    for name, validator, path in checks:
        passed, errors = validator(path)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if args.verbose:
            print(f"       Path: {path}")
        for err in errors:
            print(f"    - {err}")
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("✓ All validations passed!")
        sys.exit(0)
    else:
        print("✗ Some validations failed. Please fix issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
