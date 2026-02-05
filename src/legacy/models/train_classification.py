from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.preprocessing import PreprocessorSpec, build_preprocessor


@dataclass(frozen=True)
class ClassificationModelSpec:
    key: str
    pipeline: Pipeline


def build_classification_models(
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    categorical_features_dense: list[str] | None = None,
    seed: int,
) -> list[ClassificationModelSpec]:
    sparse_pre = build_preprocessor(
        PreprocessorSpec(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            dense_output=False,
        )
    )
    dense_pre = build_preprocessor(
        PreprocessorSpec(
            numeric_features=numeric_features,
            categorical_features=(categorical_features_dense if categorical_features_dense is not None else categorical_features),
            dense_output=True,
        )
    )

    bm2 = Pipeline(
        steps=[
            ("pre", sparse_pre),
            ("model", LogisticRegression(max_iter=8000, tol=1e-3, solver="lbfgs")),
        ]
    )

    m1 = Pipeline(
        steps=[
            ("pre", sparse_pre),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=0.5,
                    max_iter=15000,  # Increased from 4000 to avoid ConvergenceWarning
                    tol=1e-3,  # Relaxed from default 1e-4
                ),
            ),
        ]
    )

    m2 = Pipeline(
        steps=[
            ("pre", dense_pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=seed,
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=250,
                ),
            ),
        ]
    )

    return [
        ClassificationModelSpec(key="BM2_metadata_logit", pipeline=bm2),
        ClassificationModelSpec(key="M1_text_logitl1", pipeline=m1),
        ClassificationModelSpec(key="M2_text_hgbc", pipeline=m2),
    ]
