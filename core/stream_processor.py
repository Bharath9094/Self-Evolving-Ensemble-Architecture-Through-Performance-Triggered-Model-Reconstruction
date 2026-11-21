"""Sequential prequential evaluation and performance-triggered reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config import (
    BURN_IN,
    CHUNK_SIZE,
    PAGE_HINKLEY_DELTA,
    PAGE_HINKLEY_THRESHOLD,
    RECONSTRUCTION_WINDOW,
    RESET_FACTOR,
    VALIDATION_FRACTION,
)
from core.ensemble import HeterogeneousEnsemble, build_pipeline, candidate_estimators
from core.page_hinkley import PageHinkley


def classification_metrics(y_true, y_pred):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    else:
        fpr = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "false_positive_rate": fpr,
        "confusion_matrix": cm.tolist(),
        "classes": [str(item) for item in labels],
    }


class StreamExperiment:
    """Keeps training and evaluation order intact; each step predicts before learning."""

    def __init__(
        self,
        X,
        y,
        *,
        dataset_name="Dataset",
        chunk_size=CHUNK_SIZE,
        burn_in=BURN_IN,
        window_size=RECONSTRUCTION_WINDOW,
        alpha=0.7,
        eta=10.0,
        delta=PAGE_HINKLEY_DELTA,
        threshold=PAGE_HINKLEY_THRESHOLD,
        reset_factor=RESET_FACTOR,
        random_seed=42,
    ):
        if len(X) != len(y) or len(X) < 20:
            raise ValueError("X and y must have equal lengths and at least 20 rows.")
        if chunk_size < 1 or window_size < 10:
            raise ValueError("Chunk size must be positive and reconstruction window at least 10.")
        self.X = X.reset_index(drop=True).copy()
        self.y = pd.Series(y).reset_index(drop=True)
        self.dataset_name = dataset_name
        self.chunk_size = int(chunk_size)
        self.window_size = int(window_size)
        self.alpha, self.eta, self.reset_factor = alpha, eta, reset_factor
        self.delta, self.threshold, self.random_seed = delta, threshold, random_seed
        self.initial_size = min(int(burn_in), max(10, int(len(X) * 0.7)))
        self.initial_size = min(self.initial_size, len(X) - 1)
        if self.initial_size < 7:
            raise ValueError("Not enough initial training rows; provide at least 20 samples.")
        self.model = HeterogeneousEnsemble(alpha, eta, random_seed)
        self.model.fit(self.X.iloc[:self.initial_size], self.y.iloc[:self.initial_size])
        self.position = self.initial_size
        self.chunk_number = 0
        self.records = []
        self.prediction_rows = []
        self.reconstruction_log = []
        self.events = [{
            "position": self.initial_size,
            "event": "Initial ensemble trained",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
        self.member_monitors = [PageHinkley(delta, threshold) for _ in self.model.members]
        self.ensemble_monitor = PageHinkley(delta, threshold)
        self.monitor_records = []
        self.elapsed_seconds = 0.0
        self.processed_labels = []
        self.processed_predictions = []
        self.status = "ready"

    @property
    def complete(self):
        return self.position >= len(self.X)

    @property
    def processed(self):
        return self.position - self.initial_size

    def step(self):
        if self.complete:
            self.status = "complete"
            return None
        start = self.position
        end = min(start + self.chunk_size, len(self.X))
        X_chunk, y_chunk = self.X.iloc[start:end], self.y.iloc[start:end]
        began = perf_counter()
        member_preds = self.model.member_predictions(X_chunk)
        ensemble_pred = self.model.predict(X_chunk)
        chunk_errors = [np.asarray(pred) != y_chunk.to_numpy() for pred in member_preds]
        ensemble_errors = np.asarray(ensemble_pred) != y_chunk.to_numpy()

        member_alarms = set()
        ensemble_alarm = False
        for offset in range(len(y_chunk)):
            for idx, monitor in enumerate(self.member_monitors):
                if monitor.update(float(chunk_errors[idx][offset])):
                    member_alarms.add(idx)
            if self.ensemble_monitor.update(float(ensemble_errors[offset])):
                ensemble_alarm = True
            self.monitor_records.append({
                "position": start + offset + 1,
                "ensemble_statistic": self.ensemble_monitor.statistic,
                "ensemble_mean_error": self.ensemble_monitor.mean,
                "ensemble_alarm": ensemble_alarm,
                **{
                    f"member_{i + 1}_alarm": i in member_alarms
                    for i in range(len(self.member_monitors))
                },
                **{
                    f"member_{i + 1}_statistic": monitor.statistic
                    for i, monitor in enumerate(self.member_monitors)
                },
            })

        self.model.update_errors(y_chunk, member_preds)
        self.position = end
        self.chunk_number += 1
        self.processed_labels.extend(y_chunk.tolist())
        self.processed_predictions.extend(np.asarray(ensemble_pred).tolist())
        for offset, (truth, pred) in enumerate(zip(y_chunk, ensemble_pred)):
            self.prediction_rows.append({
                "position": start + offset,
                "actual": truth,
                "prediction": pred,
                "correct": bool(truth == pred),
            })
        metric = classification_metrics(y_chunk, ensemble_pred)
        self.records.append({
            "chunk": self.chunk_number,
            "position": end,
            "instances": len(y_chunk),
            "accuracy": metric["accuracy"],
            "f1_macro": metric["f1_macro"],
            "processing_seconds": 0.0,
            "ensemble_statistic": self.ensemble_monitor.statistic,
            "ensemble_mean_error": self.ensemble_monitor.mean,
            **{
                f"member_{i + 1}_accuracy": float(1 - np.mean(errors))
                for i, errors in enumerate(chunk_errors)
            },
        })

        targets = set(member_alarms)
        if ensemble_alarm:
            targets.update(sorted(
                range(len(self.model.members)),
                key=lambda idx: self.model.members[idx].smoothed_error,
                reverse=True,
            )[:max(1, len(self.model.members) // 2)])
        if member_alarms:
            for idx in sorted(member_alarms):
                self.events.append({
                    "position": end,
                    "event": f"Page-Hinkley alarm: {self.model.members[idx].name} (member {idx + 1})",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        if ensemble_alarm:
            self.events.append({
                "position": end,
                "event": "Page-Hinkley alarm: ensemble",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        for idx in sorted(targets):
            self._reconstruct(idx, "Ensemble-level alarm" if ensemble_alarm else "Member-level Page-Hinkley alarm")
        for i, member in enumerate(self.model.members):
            self.records[-1][f"member_{i + 1}_weight"] = member.weight
        duration = perf_counter() - began
        self.records[-1]["processing_seconds"] = duration
        self.elapsed_seconds += duration
        if not targets and self.chunk_number == 1:
            self.events.append({
                "position": end,
                "event": "Incoming chunk evaluated (no reconstruction alarm)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        self.status = "complete" if self.complete else "running"
        return self.records[-1]

    def run_to_completion(self):
        while not self.complete:
            self.step()
        return self.summary()

    def _reconstruct(self, member_index, reason):
        end = self.position
        first = max(0, end - self.window_size)
        X_window, y_window = self.X.iloc[first:end], self.y.iloc[first:end]
        if len(X_window) < 10 or y_window.nunique() < 2:
            self.events.append({
                "position": end,
                "event": f"Reconstruction skipped for member {member_index + 1}: insufficient window/classes",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self.member_monitors[member_index].reset()
            return
        split_at = max(2, int(len(X_window) * (1 - VALIDATION_FRACTION)))
        split_at = min(split_at, len(X_window) - 1)
        X_train, y_train = X_window.iloc[:split_at], y_window.iloc[:split_at]
        X_valid, y_valid = X_window.iloc[split_at:], y_window.iloc[split_at:]
        if y_train.nunique() < 2 or y_valid.empty:
            self.member_monitors[member_index].reset()
            return

        previous = self.model.members[member_index]
        previous_name = previous.name
        previous_error = previous.smoothed_error
        start_time = perf_counter()
        validation_results = []
        for name, candidate in candidate_estimators(self.random_seed, len(X_train)):
            try:
                estimator = build_pipeline(X_train, candidate, self.random_seed)
                estimator.fit(X_train, y_train)
                error = float(np.mean(estimator.predict(X_valid) != y_valid.to_numpy()))
                validation_results.append((error, name, candidate))
            except (ValueError, TypeError) as exc:
                self.events.append({
                    "position": end,
                    "event": f"Candidate {name} unavailable during reconstruction: {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        if not validation_results:
            self.member_monitors[member_index].reset()
            raise RuntimeError("Every candidate learner failed during reconstruction.")
        validation_error, new_name, selected = min(validation_results, key=lambda row: row[0])
        final_estimator = build_pipeline(X_window, selected, self.random_seed)
        final_estimator.fit(X_window, y_window)
        previous_model_name = previous.estimator.named_steps["model"].__class__.__name__
        self.model.members[member_index].name = new_name
        self.model.members[member_index].estimator = final_estimator
        self.model.members[member_index].smoothed_error = previous_error * self.reset_factor
        self.model._normalize_weights()
        self.member_monitors[member_index].reset()
        record = {
            "reconstruction_id": len(self.reconstruction_log) + 1,
            "position": end,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "member": member_index + 1,
            "previous_learner": previous_name,
            "new_learner": new_name,
            "previous_performance": 1 - previous_error,
            "validation_performance": 1 - validation_error,
            "validation_error": validation_error,
            "reason": reason,
            "window_size": len(X_window),
            "duration_seconds": perf_counter() - start_time,
            "candidate_results": [
                {"learner": name, "validation_error": error}
                for error, name, _ in sorted(validation_results)
            ],
        }
        self.reconstruction_log.append(record)
        self.events.append({
            "position": end,
            "event": f"Member {member_index + 1} reconstructed: {previous_model_name} -> {new_name}",
            "timestamp": record["timestamp"],
        })
        if reason == "Ensemble-level alarm":
            self.ensemble_monitor.reset()

    def summary(self):
        metrics = classification_metrics(self.processed_labels, self.processed_predictions) if self.processed_labels else {}
        return {
            "dataset_name": self.dataset_name,
            "initial_training_rows": self.initial_size,
            "instances_evaluated": self.processed,
            "chunks": self.chunk_number,
            "reconstruction_count": len(self.reconstruction_log),
            "processing_seconds": self.elapsed_seconds,
            "status": self.status,
            **metrics,
        }

    def results_frame(self):
        return pd.DataFrame(self.records)

    def predictions_frame(self):
        return pd.DataFrame(self.prediction_rows)

    def reconstruction_frame(self):
        return pd.DataFrame(self.reconstruction_log)

    def monitors_frame(self):
        return pd.DataFrame(self.monitor_records)
