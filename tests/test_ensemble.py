import numpy as np
import pandas as pd

from core.ensemble import HeterogeneousEnsemble
from data.preprocessing import make_preprocessor


def test_five_learners_train_and_weighted_predictions_are_valid():
    X = pd.DataFrame({
        "number": np.arange(60, dtype=float),
        "category": ["a", "b", "c"] * 20,
    })
    y = pd.Series([0, 1] * 30)
    model = HeterogeneousEnsemble(random_seed=7).fit(X, y)
    probabilities = model.predict_proba(X.iloc[:8])
    assert len(model.members) == 5
    assert probabilities.shape == (8, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert np.isclose(sum(member.weight for member in model.members), 1)
    assert len(model.predict(X.iloc[:8])) == 8


def test_preprocessor_handles_missing_mixed_features():
    X = pd.DataFrame({"n": [1.0, None, 3.0], "c": ["x", None, "y"]})
    transformer = make_preprocessor(X)
    transformed = transformer.fit_transform(X)
    assert transformed.shape[0] == 3
    assert np.isfinite(transformed).all()
