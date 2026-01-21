from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_onehot_encoder(*, dense: bool) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    try:
        kwargs["sparse_output"] = not dense
        return OneHotEncoder(**kwargs)
    except TypeError:
        kwargs["sparse"] = not dense
        return OneHotEncoder(**kwargs)


@dataclass(frozen=True)
class PreprocessorSpec:
    numeric_features: list[str]
    categorical_features: list[str]
    dense_output: bool


def build_preprocessor(spec: PreprocessorSpec) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder(dense=spec.dense_output)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, spec.numeric_features),
            ("cat", cat_pipe, spec.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

