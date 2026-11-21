import numpy as np
import pandas as pd

from core.ensemble import candidate_estimators
from core.baselines import run_comparison
from core.stream_processor import StreamExperiment


def make_data(rows=120):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "x1": rng.normal(size=rows),
        "x2": rng.normal(size=rows),
    })
    y = pd.Series((X.x1 + X.x2 > 0).astype(int))
    return X, y


def test_stream_predicts_only_evaluation_rows_and_preserves_order():
    X, y = make_data()
    experiment = StreamExperiment(X, y, burn_in=70, chunk_size=13)
    initial_position = experiment.position
    record = experiment.step()
    assert record["position"] == initial_position + 13
    assert experiment.predictions_frame()["position"].tolist() == list(range(initial_position, initial_position + 13))
    assert experiment.processed == 13


def test_reconstruction_selects_and_trains_a_real_candidate():
    X, y = make_data()
    experiment = StreamExperiment(X, y, burn_in=70, window_size=60)
    experiment._reconstruct(0, "test alarm")
    assert len(experiment.reconstruction_log) == 1
    record = experiment.reconstruction_log[0]
    assert record["new_learner"] in {name for name, _ in candidate_estimators(train_size=42)}
    assert record["window_size"] == 60
    assert record["validation_performance"] >= 0
    assert experiment.model.members[0].estimator.predict(X.iloc[:3]).shape == (3,)


def test_end_to_end_metrics_are_from_predictions():
    X, y = make_data()
    experiment = StreamExperiment(X, y, burn_in=70, chunk_size=20)
    result = experiment.run_to_completion()
    assert result["instances_evaluated"] == 50
    assert 0 <= result["accuracy"] <= 1
    assert result["confusion_matrix"]


def test_page_hinkley_alarm_runs_reconstruction_without_duplicate_members():
    X, y = make_data()
    y.iloc[60:] = 1 - y.iloc[60:]
    experiment = StreamExperiment(
        X, y, burn_in=60, chunk_size=20, window_size=40,
        delta=0.01, threshold=0.01,
    )
    experiment.step()
    assert any("Page-Hinkley alarm" in event["event"] for event in experiment.events)
    positions = [record["position"] for record in experiment.reconstruction_log]
    members = [record["member"] for record in experiment.reconstruction_log]
    assert positions
    assert len(set(members)) == len(members)
    assert all(record["new_learner"] for record in experiment.reconstruction_log)


def test_baseline_comparison_uses_the_same_ordered_evaluation_window():
    X, y = make_data(100)
    summary, outcomes = run_comparison(
        X, y, chunk_size=20, burn_in=70, window_size=40, random_seed=42
    )
    assert set(summary["Method"]) == {
        "SEE-PMR", "Static Ensemble", "Periodic Ensemble",
        "DWM-style Ensemble", "Window-DT",
    }
    assert all(result["instances"] == 30 for result in outcomes.values())
    assert outcomes["Window-DT"]["reconstructions"] == 2
