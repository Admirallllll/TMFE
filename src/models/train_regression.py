from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.models.preprocessing import PreprocessorSpec, build_preprocessor


@dataclass(frozen=True)
class RegressionModelSpec:
    key: str
    pipeline: Pipeline


def build_regression_models(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_features_dense: list[str] | None = None,
    seed: int,
) -> list[RegressionModelSpec]:
    meta_pre = build_preprocessor(
        PreprocessorSpec(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            dense_output=False,
        )
    )
    dense_cats = categorical_features_dense if categorical_features_dense is not None else categorical_features
    dense_pre = build_preprocessor(
        PreprocessorSpec(
            numeric_features=numeric_features,
            categorical_features=dense_cats,
            dense_output=True,
        )
    )

    bm2 = Pipeline(
        steps=[
            ("pre", meta_pre),
            ("model", Ridge(alpha=5.0, random_state=seed)),
        ]
    )

    m1 = Pipeline(
        steps=[
            ("pre", meta_pre),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9],
                    alphas=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                    cv=TimeSeriesSplit(n_splits=3),
                    max_iter=50000,  # Increased from 20000 to avoid ConvergenceWarning
                    tol=1e-3,  # Relaxed from default 1e-4
                    random_state=seed,
                    n_jobs=-1,  # Enable parallelization
                ),
            ),
        ]
    )

    m2 = Pipeline(
        steps=[
            ("pre", dense_pre),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=seed,
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=250,
                ),
            ),
        ]
    )

    return [
        RegressionModelSpec(key="BM2_metadata_ridge", pipeline=bm2),
        RegressionModelSpec(key="M1_text_elasticnet", pipeline=m1),
        RegressionModelSpec(key="M2_text_hgbr", pipeline=m2),
    ]
