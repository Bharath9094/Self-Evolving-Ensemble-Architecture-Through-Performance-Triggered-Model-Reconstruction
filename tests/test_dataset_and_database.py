import json

import pandas as pd
import pytest

from data.dataset_generator import generate_dataset
from database.experiment_db import ExperimentDB


def test_synthetic_dataset_is_reproducible_and_ordered():
    first, target = generate_dataset("Recurring Concept Drift Dataset", 100, 4, random_seed=9)
    second, _ = generate_dataset("Recurring Concept Drift Dataset", 100, 4, random_seed=9)
    pd.testing.assert_frame_equal(first, second)
    assert target == "target"
    assert len(first) == 100


def test_synthetic_dataset_rejects_invalid_size():
    with pytest.raises(ValueError):
        generate_dataset("SEA", n_samples=10)


def test_experiment_database_persists_without_overwriting(tmp_path):
    db = ExperimentDB(tmp_path / "experiments.sqlite3")
    db.save("first", "demo", {"seed": 1}, {"accuracy": 0.8}, [])
    assert json.loads(db.list()[0]["summary"])["accuracy"] == 0.8
    with pytest.raises(Exception):
        db.save("first", "demo", {}, {}, [])

