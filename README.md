# SEE-PMR — Self-Evolving Ensemble Learning System

SEE-PMR is a local Streamlit research application for ordered, prequential
classification experiments. It trains a five-member heterogeneous ensemble,
tracks each member and the ensemble with Page-Hinkley monitors, and selects
replacement learners from a recent temporally split validation window after
an alarm.

## Features

- CSV, XLSX, and XLS dataset upload and profiling.
- Reproducible SEA-style, hyperplane, recurring, and gradual drift generators.
- Numerical imputation/scaling and categorical imputation/one-hot encoding.
- Five learners: depth-3 and depth-6 decision trees, Gaussian Naive Bayes,
  logistic regression, and K-nearest neighbors.
- Weighted probabilistic predictions and exponentially smoothed member errors.
- Ordered test-then-train chunk evaluation with configurable Page-Hinkley
  monitoring and actual validation-guided reconstruction.
- Static, periodic, simplified DWM-style, and windowed decision-tree baselines.
- Plotly metrics, persistent SQLite experiment history, CSV/model downloads,
  and PDF experiment reports.

## Installation

Python 3.11 or newer is recommended. From this folder, run:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`).

## Running an experiment

1. In **Dataset Management**, upload a labeled classification file or create
   one of the reproducible synthetic streams. Select the target and features.
2. Review **Data Preprocessing** and adjust optional experiment parameters in
   **Ensemble Configuration**.
3. Use **Model Training** to fit the five learners on the initial ordered
   segment. By default, up to 500 records are used for initial training; for
   smaller datasets, the split adapts while preserving at least 30% for
   evaluation.
4. In **Live Stream Simulation**, process one chunk at a time or evaluate the
   remaining stream. Predictions are made before labels from that chunk are
   used to update monitors and learner weights.
5. Review alarms, actual reconstructions, metrics, and baseline comparisons.
   Save named experiment summaries in **Experiment History**.

Experiments, not uploaded dataset contents, are persisted in
`outputs/experiments.sqlite3`. Names are unique; saving an existing name fails
instead of replacing its record.

## Default parameters

| Parameter | Default |
|---|---:|
| Ensemble size | 5 |
| Smoothing factor | 0.7 |
| Weight sharpness | 10 |
| Reconstruction window | 250 |
| Chunk size | 50 |
| Burn-in | 500 |
| Page-Hinkley delta | 0.03 |
| Page-Hinkley threshold | 0.25 |
| Reset factor | 0.8 |
| Candidate validation split | 70% train / 30% validation |

Stream order is retained. Preprocessing is fitted on each training partition,
not the full stream. Categorical and multiclass targets are supported.
Page-Hinkley monitors are updated with per-instance prequential errors.
The DWM-style method is intentionally a simplified research baseline, not a
claim of implementing the complete published algorithm. Uploaded data is
static and is not described as an external live source. Synthetic drift is
artificial by design.

## Project layout

```text
app.py                  Streamlit UI and workflow
config.py               Research defaults
core/                   Ensemble, drift monitor, reconstruction, baselines
data/                   Dataset readers, synthetic generators, preprocessing
database/                SQLite experiment persistence
visualization/           Plotly chart helpers
assets/                  Dashboard styling
tests/                   Focused algorithm and persistence tests
outputs/                 Local experiment database (created at runtime)
```

## Tests

```powershell
python -m pytest -q
```

## Limitations

- The simulator consumes a local file or generated frame in source order; it
  does not connect to a real-time data broker.
- The displayed drift point in synthetic data represents the generator's
  configured change, not a guaranteed Page-Hinkley alarm.
- Reconstruction is skipped when the recent window is too small or contains
  only one target class. Candidate training failures are surfaced in the
  experiment event log; reconstruction fails explicitly if all candidates
  are unavailable.
- PDF creation requires the `reportlab` dependency included in
  `requirements.txt`.
