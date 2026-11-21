"""Comparable prequential baselines for the same ordered stream."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from core.ensemble import HeterogeneousEnsemble, build_pipeline, candidate_estimators


def run_comparison(X, y, *, chunk_size=50, burn_in=500, window_size=250, random_seed=42):
    """Run SEE-PMR and four baselines over identical, ordered evaluation chunks."""
    X, y = X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    if len(X) < 20:
        raise ValueError("At least 20 rows are required for comparison.")
    initial = min(int(burn_in), max(10, int(len(X) * 0.7)), len(X) - 1)
    if initial < 7:
        raise ValueError("Not enough rows to initialize models.")
    methods = ["SEE-PMR", "Static Ensemble", "Periodic Ensemble", "DWM-style Ensemble", "Window-DT"]
    outcomes = {}
    summary = []
    for method in methods:
        started = perf_counter()
        if method == "SEE-PMR":
            from core.stream_processor import StreamExperiment

            experiment = StreamExperiment(
                X, y, dataset_name="comparison", chunk_size=chunk_size,
                burn_in=burn_in, window_size=window_size,
                random_seed=random_seed,
            )
            experiment.run_to_completion()
            duration = perf_counter() - started
            result = experiment.summary()
            scores = experiment.results_frame().get("accuracy", pd.Series(dtype=float)).tolist()
            outcomes[method] = {
                "accuracy_by_chunk": scores,
                "accuracy": result.get("accuracy", 0.0),
                "f1_macro": result.get("f1_macro", 0.0),
                "reconstructions": result.get("reconstruction_count", 0),
                "processing_seconds": duration,
                "instances": result.get("instances_evaluated", 0),
            }
            summary.append({
                "Method": method,
                "Mean prequential accuracy": outcomes[method]["accuracy"],
                "Macro F1": outcomes[method]["f1_macro"],
                "Reconstructed members": outcomes[method]["reconstructions"],
                "Processing seconds": duration,
                "Accuracy per reconstruction": outcomes[method]["accuracy"] / max(1, outcomes[method]["reconstructions"]),
            })
            continue
        if method == "Window-DT":
            from core.ensemble import build_pipeline

            initial_model = build_pipeline(
                X.iloc[:initial],
                DecisionTreeClassifier(max_depth=6, random_state=random_seed),
                random_seed,
            )
            initial_model.fit(X.iloc[:initial], y.iloc[:initial])
            labels, preds, chunk_scores = [], [], []
            reconstructions = 0
            for start in range(initial, len(X), chunk_size):
                end = min(start + chunk_size, len(X))
                xb, yb = X.iloc[start:end], y.iloc[start:end]
                prediction = initial_model.predict(xb)
                labels.extend(yb.tolist())
                preds.extend(prediction.tolist())
                chunk_scores.append(accuracy_score(yb, prediction))
                recent_start = max(0, end - 500)
                initial_model = build_pipeline(
                    X.iloc[recent_start:end],
                    DecisionTreeClassifier(max_depth=6, random_state=random_seed),
                    random_seed,
                )
                initial_model.fit(X.iloc[recent_start:end], y.iloc[recent_start:end])
                reconstructions += 1
            duration = perf_counter() - started
            accuracy = float(accuracy_score(labels, preds))
            f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
            outcomes[method] = {
                "accuracy_by_chunk": chunk_scores,
                "accuracy": accuracy,
                "f1_macro": f1,
                "reconstructions": reconstructions,
                "processing_seconds": duration,
                "instances": len(labels),
            }
            summary.append({
                "Method": method,
                "Mean prequential accuracy": accuracy,
                "Macro F1": f1,
                "Reconstructed members": reconstructions,
                "Processing seconds": duration,
                "Accuracy per reconstruction": accuracy / max(1, reconstructions),
            })
            continue
        model = HeterogeneousEnsemble(random_seed=random_seed).fit(X.iloc[:initial], y.iloc[:initial])
        weights = np.full(5, 0.2)
        labels, preds = [], []
        reconstructions, chunk_scores = 0, []
        chunk_no = 0
        for start in range(initial, len(X), chunk_size):
            end = min(start + chunk_size, len(X))
            xb, yb = X.iloc[start:end], y.iloc[start:end]
            member_preds = model.member_predictions(xb)
            if method == "Window-DT":
                model_pred = member_preds[1]
            elif method == "DWM-style Ensemble":
                weighted = np.zeros((len(xb), len(model.classes_)))
                for weight, pred in zip(weights, member_preds):
                    for col, label in enumerate(model.classes_):
                        weighted[:, col] += weight * (np.asarray(pred) == label)
                model_pred = model.classes_[np.argmax(weighted, axis=1)]
            else:
                model_pred = model.predict(xb)
            labels.extend(yb.tolist())
            preds.extend(np.asarray(model_pred).tolist())
            chunk_error = float(np.mean(np.asarray(model_pred) != yb.to_numpy()))
            chunk_no += 1
            chunk_scores.append(accuracy_score(yb, model_pred))
            model.update_errors(yb, member_preds)

            end_train = end
            baseline_window = 500 if method == "Periodic Ensemble" else max(window_size, 500)
            first = max(0, end_train - baseline_window)
            if method == "Periodic Ensemble" and chunk_no % 10 == 0:
                idx = int(np.argmax([m.smoothed_error for m in model.members]))
                _replace(model, idx, X.iloc[first:end_train], y.iloc[first:end_train], random_seed)
                reconstructions += 1
            elif method == "DWM-style Ensemble":
                weights *= np.where(np.asarray([
                    np.mean(np.asarray(p) != yb.to_numpy()) for p in member_preds
                ]) > 0.30, 0.5, 1.0)
                if chunk_error > 0.30:
                    idx = int(np.argmin(weights))
                    _replace(model, idx, X.iloc[max(0, end_train - 250):end_train], y.iloc[max(0, end_train - 250):end_train], random_seed)
                    weights[idx] = max(weights.mean(), 1e-8)
                    reconstructions += 1
        elapsed = perf_counter() - started
        accuracy = float(accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        outcomes[method] = {
            "accuracy_by_chunk": chunk_scores,
            "accuracy": accuracy,
            "f1_macro": f1,
            "reconstructions": reconstructions,
            "processing_seconds": elapsed,
            "instances": len(labels),
        }
        summary.append({
            "Method": method,
            "Mean prequential accuracy": accuracy,
            "Macro F1": f1,
            "Reconstructed members": reconstructions,
            "Processing seconds": elapsed,
            "Accuracy per reconstruction": accuracy / max(1, reconstructions),
        })
    return pd.DataFrame(summary), outcomes


def _replace(ensemble, member_index, X_window, y_window, random_seed):
    if len(X_window) < 5 or y_window.nunique() < 2:
        return
    candidates = candidate_estimators(random_seed, len(X_window))
    name, estimator = candidates[member_index % len(candidates)]
    fitted = build_pipeline(X_window, estimator, random_seed)
    fitted.fit(X_window, y_window)
    ensemble.members[member_index].name = name
    ensemble.members[member_index].estimator = fitted
    ensemble._normalize_weights()
