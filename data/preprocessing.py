"""Leakage-safe preprocessing pipeline construction."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric)
        )
    if categorical:
        transformers.append(
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical)
        )
    if not transformers:
        raise ValueError("Select at least one usable feature.")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
