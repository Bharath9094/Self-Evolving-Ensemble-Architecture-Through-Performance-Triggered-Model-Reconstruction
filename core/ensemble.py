"""Heterogeneous probabilistic ensemble and its five candidate learners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from data.preprocessing import make_preprocessor


@dataclass
class Member:
    name: str
    estimator: Pipeline
    smoothed_error: float = 0.5
    weight: float = 0.2


def candidate_estimators(random_seed: int = 42, train_size: int = 250):
    return [
        ("Decision Tree (depth 3)", DecisionTreeClassifier(max_depth=3, random_state=random_seed)),
        ("Decision Tree (depth 6)", DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=random_seed)),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=random_seed)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=max(1, min(7, train_size)))),
    ]


def build_pipeline(X, model, random_seed: int = 42):
    return Pipeline([("preprocess", make_preprocessor(X)), ("model", clone(model))])


class HeterogeneousEnsemble:
    def __init__(self, alpha=0.7, eta=10.0, random_seed=42):
        if not 0 < alpha <= 1 or eta <= 0:
            raise ValueError("alpha must be in (0, 1] and eta must be positive.")
        self.alpha = alpha
        self.eta = eta
        self.random_seed = random_seed
        self.members: list[Member] = []
        self.classes_ = None

    def fit(self, X, y):
        if len(X) < 2:
            raise ValueError("At least two training rows are required.")
        self.classes_ = np.unique(y)
        models = candidate_estimators(self.random_seed, len(X))
        members = []
        for name, model in models:
            estimator = build_pipeline(X, model, self.random_seed)
            estimator.fit(X, y)
            members.append(Member(name, estimator))
        self.members = members
        self._normalize_weights()
        return self

    def _normalize_weights(self):
        if not self.members:
            return
        raw = np.asarray([np.exp(-self.eta * m.smoothed_error) for m in self.members])
        if not np.all(np.isfinite(raw)) or raw.sum() == 0:
            raw = np.ones(len(self.members))
        raw /= raw.sum()
        for member, weight in zip(self.members, raw):
            member.weight = float(weight)

    def member_predictions(self, X):
        if not self.members:
            raise RuntimeError("The ensemble has not been trained.")
        return [member.estimator.predict(X) for member in self.members]

    def predict_proba(self, X):
        if not self.members:
            raise RuntimeError("The ensemble has not been trained.")
        combined = np.zeros((len(X), len(self.classes_)), dtype=float)
        for member in self.members:
            estimator = member.estimator
            if hasattr(estimator, "predict_proba"):
                probabilities = estimator.predict_proba(X)
                model_classes = estimator.named_steps["model"].classes_
                for column, label in enumerate(model_classes):
                    target_column = np.flatnonzero(self.classes_ == label)
                    if len(target_column):
                        combined[:, target_column[0]] += member.weight * probabilities[:, column]
            else:
                pred = estimator.predict(X)
                for column, label in enumerate(self.classes_):
                    combined[:, column] += member.weight * (pred == label)
        totals = combined.sum(axis=1, keepdims=True)
        return np.divide(combined, totals, out=np.full_like(combined, 1 / len(self.classes_)), where=totals > 0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def update_errors(self, y, member_predictions):
        y = np.asarray(y)
        for member, predictions in zip(self.members, member_predictions):
            error = float(np.mean(np.asarray(predictions) != y))
            member.smoothed_error = self.alpha * member.smoothed_error + (1 - self.alpha) * error
        self._normalize_weights()
