"""SEE-PMR: Streamlit research application."""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from config import (
    BURN_IN, CHUNK_SIZE, PAGE_HINKLEY_DELTA, PAGE_HINKLEY_THRESHOLD,
    RECONSTRUCTION_WINDOW, RESET_FACTOR, SMOOTHING_ALPHA, VALIDATION_FRACTION,
    WEIGHT_SHARPNESS,
)
from core.baselines import run_comparison
from core.stream_processor import StreamExperiment, classification_metrics
from data.dataset_generator import generate_dataset
from data.dataset_loader import read_dataset
from data.preprocessing import make_preprocessor
from database.experiment_db import ExperimentDB
from visualization.charts import (
    accuracy_chart, comparison_chart, member_accuracy_chart, monitor_chart,
    accuracy_cost_chart, reconstruction_comparison_chart,
    reconstruction_count_chart, reconstruction_timeline, weight_chart,
    weight_history_chart,
)


st.set_page_config(
    page_title="SEE-PMR | Adaptive Ensemble Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSET_CSS = Path(__file__).parent / "assets" / "style.css"
if ASSET_CSS.exists():
    st.markdown(ASSET_CSS.read_text(encoding="utf-8"), unsafe_allow_html=True)

PAGES = [
    "Overview Dashboard", "Dataset Management", "Data Preprocessing",
    "Ensemble Configuration", "Model Training", "Live Stream Simulation",
    "Drift Detection", "Model Reconstruction", "Model Performance",
    "Algorithm Comparison", "Experiment History", "Reports and Downloads",
    "About SEE-PMR",
]


def init_state():
    defaults = {
        "dataset": None,
        "dataset_name": None,
        "target": None,
        "features": None,
        "experiment": None,
        "comparison": None,
        "comparison_outcomes": None,
        "events": [],
        "simulation_status": "stopped",
        "preprocessed": None,
        "show_dataset_preview": False,
        "config": {
            "chunk_size": CHUNK_SIZE, "burn_in": BURN_IN,
            "window_size": RECONSTRUCTION_WINDOW,
            "alpha": SMOOTHING_ALPHA, "eta": WEIGHT_SHARPNESS,
            "delta": PAGE_HINKLEY_DELTA,
            "threshold": PAGE_HINKLEY_THRESHOLD,
            "reset_factor": RESET_FACTOR, "random_seed": 42,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_dataset(frame, name, target=None):
    if frame.empty:
        raise ValueError("The selected dataset is empty.")
    frame = frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Dataset columns must have unique names.")
    st.session_state.dataset = frame.reset_index(drop=True)
    st.session_state.dataset_name = name
    st.session_state.target = target if target in frame.columns else frame.columns[-1]
    st.session_state.features = [column for column in frame.columns if column != st.session_state.target]
    st.session_state.experiment = None
    st.session_state.preprocessed = None
    st.session_state.show_dataset_preview = False
    st.session_state.comparison = None
    st.session_state.events = [{"event": f"Dataset loaded: {name}", "position": 0}]


def load_uploaded(upload):
    return read_dataset(upload, upload.name)


def selected_data():
    frame = st.session_state.dataset
    target = st.session_state.target
    features = st.session_state.features
    if frame is None:
        raise ValueError("Load a dataset from Dataset Management first.")
    if target not in frame.columns:
        raise ValueError("The selected target column is missing.")
    if not features or any(feature not in frame.columns for feature in features):
        raise ValueError("Select at least one valid feature column.")
    usable = frame.loc[:, features + [target]].dropna(subset=[target]).reset_index(drop=True)
    if len(usable) < 20:
        raise ValueError("At least 20 labeled rows are required for an experiment.")
    if usable[target].nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    return usable[features], usable[target], usable


def active_experiment():
    exp = st.session_state.experiment
    if exp is None:
        st.info("Train an ensemble or start a stream experiment to populate this view.")
    return exp


def metrics_cards(exp):
    summary = exp.summary()
    average_reconstruction = (
        float(np.mean([item["duration_seconds"] for item in exp.reconstruction_log]))
        if exp.reconstruction_log else 0.0
    )
    columns = st.columns(6)
    columns[0].metric("Prequential accuracy", f"{summary.get('accuracy', 0):.1%}")
    columns[1].metric("Active models", len(exp.model.members))
    alarm_positions = {
        event["position"] for event in exp.events
        if "alarm" in event["event"].lower()
    }
    columns[2].metric("Drift events", len(alarm_positions))
    columns[3].metric("Reconstructions", len(exp.reconstruction_log))
    columns[4].metric("Avg. reconstruction time", f"{average_reconstruction:.3f}s")
    columns[5].metric("Ensemble status", exp.status.title())


def build_pdf(exp):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from matplotlib.figure import Figure

    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter)
    summary = exp.summary()
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SEE-PMR Experiment Summary", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Dataset: {exp.dataset_name}", styles["BodyText"]),
        Paragraph(f"Training rows: {exp.initial_size}; evaluated rows: {exp.processed}.",
                  styles["BodyText"]),
        Paragraph("Methodology: ordered test-then-train prequential evaluation, "
                  "weighted heterogeneous ensemble, and per-member Page-Hinkley monitoring.",
                  styles["BodyText"]),
        Paragraph(
            f"Page-Hinkley parameters: delta={exp.delta}, threshold={exp.threshold}. "
            "Candidate selection uses a temporal 70/30 split of the recent reconstruction window.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]
    rows = [["Metric", "Value"]]
    for key in ("initial_training_rows", "instances_evaluated", "accuracy", "f1_macro",
                "reconstruction_count", "processing_seconds"):
        rows.append([key.replace("_", " ").title(), str(summary.get(key, "n/a"))])
    table = Table(rows, colWidths=[220, 260])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14314d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    if exp.records:
        figure = Figure(figsize=(6.5, 2.8), tight_layout=True)
        axis = figure.subplots()
        frame = exp.results_frame()
        axis.plot(frame["position"], frame["accuracy"], marker="o", color="#168aad")
        axis.set_ylim(0, 1)
        axis.set_xlabel("Instances processed")
        axis.set_ylabel("Chunk accuracy")
        axis.set_title("Actual prequential accuracy")
        chart = io.BytesIO()
        figure.savefig(chart, format="png", dpi=140)
        chart.seek(0)
        story.extend([Spacer(1, 12), Image(chart, width=460, height=198)])
    if exp.reconstruction_log:
        story.extend([Spacer(1, 14), Paragraph("Reconstruction events", styles["Heading2"])])
        for item in exp.reconstruction_log:
            story.append(Paragraph(
                f"Position {item['position']}: {item['previous_learner']} -> {item['new_learner']}; "
                f"validation accuracy {item['validation_performance']:.3f}; reason: {item['reason']}.",
                styles["BodyText"],
            ))
            story.append(Spacer(1, 5))
    else:
        story.extend([Spacer(1, 14), Paragraph(
            "No reconstruction was triggered in this experiment.", styles["BodyText"]
        )])
    conclusion = (
        f"Final observed prequential accuracy was {summary.get('accuracy', 0):.3f}. "
        f"The run performed {summary.get('reconstruction_count', 0)} reconstruction(s)."
    )
    story.extend([Spacer(1, 12), Paragraph("Conclusion", styles["Heading2"]),
                  Paragraph(conclusion, styles["BodyText"])])
    document.build(story)
    return output.getvalue()


def sidebar():
    st.sidebar.markdown("## ◈ SEE-PMR")
    st.sidebar.caption("Adaptive Ensemble Research Lab")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Research defaults · reproducible experiments")
    if st.session_state.dataset is not None:
        st.sidebar.success(st.session_state.dataset_name)
        st.sidebar.caption(f"{len(st.session_state.dataset):,} rows · {len(st.session_state.dataset.columns)} columns")
    else:
        st.sidebar.info("No dataset loaded")
    return page


def page_dataset():
    st.title("Dataset Management")
    st.write("Upload a labeled classification dataset, or generate a reproducible concept-drift stream.")
    tab_upload, tab_synthetic = st.tabs(["Upload dataset", "Synthetic drift datasets"])
    with tab_upload:
        upload = st.file_uploader("Drop a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if upload:
            try:
                frame = load_uploaded(upload)
                if frame.empty:
                    st.error("The uploaded dataset has no rows.")
                elif st.button("Load uploaded dataset", type="primary"):
                    set_dataset(frame, upload.name)
                    st.success("Dataset loaded. Choose the target and feature columns below.")
            except Exception as exc:
                st.error(f"Could not read dataset: {exc}")
    with tab_synthetic:
        with st.form("synthetic"):
            kind = st.selectbox("Dataset", [
                "SEA Concept Drift Dataset", "Hyperplane Concept Drift Dataset",
                "Recurring Concept Drift Dataset", "Gradual SEA Concept Drift Dataset",
            ])
            c1, c2, c3 = st.columns(3)
            samples = c1.number_input("Samples", min_value=100, max_value=20000, value=1500, step=100)
            n_features = c2.number_input("Features", min_value=2, max_value=50, value=6)
            noise = c3.slider("Noise percentage", 0, 50, 5) / 100
            c4, c5 = st.columns(2)
            drift_at = c4.slider("Drift location", 5, 95, 50) / 100
            severity = c5.slider("Drift severity", 0.0, 2.0, 1.0, 0.1)
            seed = st.number_input("Random seed", min_value=0, value=42)
            submitted = st.form_submit_button("Generate and load sample dataset", type="primary")
        if submitted:
            try:
                frame, target = generate_dataset(
                    kind, int(samples), int(n_features), noise, drift_at,
                    severity, int(seed),
                )
                set_dataset(frame, kind, target)
                st.success(f"Generated {len(frame):,} ordered rows with a configured {kind.lower()}.")
            except ValueError as exc:
                st.error(str(exc))
    frame = st.session_state.dataset
    if frame is not None:
        st.subheader("Dataset profile and selection")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(frame):,}")
        c2.metric("Columns", len(frame.columns))
        c3.metric("Duplicate rows", int(frame.duplicated().sum()))
        numerical = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [c for c in frame.columns if c not in numerical]
        c1, c2 = st.columns(2)
        c1.write(f"Numerical ({len(numerical)}): {', '.join(map(str, numerical)) or 'None'}")
        c2.write(f"Categorical ({len(categorical)}): {', '.join(map(str, categorical)) or 'None'}")
        missing = frame.isna().sum()
        st.dataframe(pd.DataFrame({"dtype": frame.dtypes.astype(str), "missing": missing}), use_container_width=True)
        target = st.selectbox(
            "Target column", frame.columns,
            index=list(frame.columns).index(st.session_state.target),
            key="target_selection",
        )
        features = st.multiselect(
            "Feature columns", [c for c in frame.columns if c != target],
            default=[c for c in st.session_state.features if c != target],
            key="feature_selection",
        )
        st.session_state.target = target
        st.session_state.features = features
        st.caption("Rows remain in their current order for stream experiments; no implicit shuffle is applied.")
        if len(frame[target].dropna().unique()) == 2:
            with st.expander("Artificial drift demonstration for uploaded data"):
                st.warning("This intentionally modifies labels and must not be interpreted as naturally occurring drift.")
                location = st.slider("Artificial drift begins at row (%)", 5, 95, 50, key="upload_drift_location")
                severity = st.slider("Artificial label-flip proportion", 0.0, 1.0, 1.0, 0.05, key="upload_drift_severity")
                if st.button("Create artificial-drift copy"):
                    modified = frame.copy()
                    classes = list(modified[target].dropna().unique())
                    begin = int(len(modified) * location / 100)
                    rng = np.random.default_rng(int(st.session_state.config["random_seed"]))
                    for row in range(begin, len(modified)):
                        if rng.random() < severity:
                            current = modified.at[row, target]
                            modified.at[row, target] = classes[(classes.index(current) + 1) % len(classes)]
                    set_dataset(modified, f"{st.session_state.dataset_name} (artificial label drift)", target)
                    st.success("Created an explicitly labeled artificial-drift copy.")
        if st.button("Preview Dataset"):
            st.session_state.show_dataset_preview = True
        if st.session_state.show_dataset_preview:
            st.subheader("Dataset preview")
            st.dataframe(frame.head(20), use_container_width=True)


def page_preprocessing():
    st.title("Data Preprocessing")
    try:
        X, y, usable = selected_data()
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.write("Numerical values use median imputation and standard scaling. Categorical values use "
             "most-frequent imputation and one-hot encoding. The preview transformer is fit on the "
             "first 70% only; stream models fit their own transformers on initial training data.")
    st.metric("Rows available", len(usable))
    st.write(f"Features: {len(X.columns)} · target classes: {y.nunique()} · duplicate rows: {usable.duplicated().sum()}")
    if st.button("Preprocess data", type="primary"):
        try:
            split = max(2, int(len(X) * 0.7))
            transformer = make_preprocessor(X.iloc[:split])
            transformed = transformer.fit(X.iloc[:split]).transform(X)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            names = transformer.get_feature_names_out()
            processed = pd.DataFrame(transformed, columns=names)
            processed[st.session_state.target] = y.reset_index(drop=True)
            st.session_state.preprocessed = processed
            st.session_state.events.append({"event": "Preprocessing preview fit on ordered training partition"})
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")
    if st.session_state.preprocessed is not None:
        st.dataframe(st.session_state.preprocessed.head(20), use_container_width=True)
        st.download_button(
            "Download processed features CSV",
            st.session_state.preprocessed.to_csv(index=False).encode("utf-8"),
            file_name="processed_features.csv", mime="text/csv",
        )


def page_configuration():
    st.title("Ensemble Configuration")
    st.write("Default research values are prefilled. Changes apply to the next experiment.")
    cfg = st.session_state.config
    with st.form("config"):
        c1, c2, c3 = st.columns(3)
        cfg["chunk_size"] = c1.number_input("Chunk size B", min_value=1, max_value=1000, value=int(cfg["chunk_size"]))
        cfg["burn_in"] = c2.number_input("Burn-in", min_value=10, max_value=10000, value=int(cfg["burn_in"]))
        cfg["window_size"] = c3.number_input("Reconstruction window L", min_value=10, max_value=10000, value=int(cfg["window_size"]))
        c1, c2, c3 = st.columns(3)
        cfg["alpha"] = c1.slider("Smoothing alpha", 0.01, 1.0, float(cfg["alpha"]))
        cfg["eta"] = c2.number_input("Weight sharpness eta", min_value=0.1, max_value=50.0, value=float(cfg["eta"]))
        cfg["reset_factor"] = c3.slider("Reset factor kappa", 0.0, 1.0, float(cfg["reset_factor"]))
        c1, c2, c3 = st.columns(3)
        cfg["delta"] = c1.number_input("Page-Hinkley delta", min_value=0.0, max_value=1.0, value=float(cfg["delta"]))
        cfg["threshold"] = c2.number_input("Page-Hinkley threshold", min_value=0.001, max_value=10.0, value=float(cfg["threshold"]))
        cfg["random_seed"] = c3.number_input("Random seed", min_value=0, value=int(cfg["random_seed"]))
        if st.form_submit_button("Save configuration", type="primary"):
            st.session_state.config = dict(cfg)
            st.success("Configuration saved for the next experiment.")
    st.info("Ensemble: depth-3 tree, depth-6 tree (leaf minimum 5), Gaussian NB, Logistic Regression, and 7-neighbor KNN.")


def new_experiment():
    X, y, _ = selected_data()
    cfg = st.session_state.config
    return StreamExperiment(
        X, y, dataset_name=st.session_state.dataset_name,
        chunk_size=int(cfg["chunk_size"]), burn_in=int(cfg["burn_in"]),
        window_size=int(cfg["window_size"]), alpha=float(cfg["alpha"]),
        eta=float(cfg["eta"]), delta=float(cfg["delta"]),
        threshold=float(cfg["threshold"]), reset_factor=float(cfg["reset_factor"]),
        random_seed=int(cfg["random_seed"]),
    )


def page_training():
    st.title("Model Training")
    st.write("Train all five heterogeneous learners on the ordered initial training segment.")
    if st.button("Train ensemble", type="primary"):
        try:
            st.session_state.experiment = new_experiment()
            st.session_state.simulation_status = "ready"
            st.session_state.events.extend(st.session_state.experiment.events)
            st.success(
                f"Initial ensemble trained on {st.session_state.experiment.initial_size:,} rows. "
                "The remaining rows are reserved for test-then-train evaluation."
            )
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    exp = active_experiment()
    if exp:
        metrics_cards(exp)
        st.dataframe(pd.DataFrame([{
            "Member": i + 1, "Learner": member.name,
            "Weight": member.weight, "Smoothed error": member.smoothed_error,
        } for i, member in enumerate(exp.model.members)]), use_container_width=True)


def page_simulation():
    st.title("Live Stream Simulation")
    st.caption("This is an ordered, local dataset stream—not an external live data feed. Each action processes at most one chunk unless Run to completion is selected.")
    exp = st.session_state.experiment
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Start simulation", type="primary"):
        if exp is None:
            try:
                exp = new_experiment()
                st.session_state.experiment = exp
                st.session_state.simulation_status = "running"
                exp.step()
            except Exception as exc:
                st.error(f"Could not start simulation: {exc}")
        elif exp.complete:
            st.warning("This stream is complete. Reset the experiment to start again.")
        else:
            st.session_state.simulation_status = "running"
            exp.step()
    if c2.button("Pause"):
        st.session_state.simulation_status = "paused"
    if c3.button("Resume"):
        if exp is None:
            st.warning("Train an ensemble first.")
        elif not exp.complete:
            st.session_state.simulation_status = "running"
            exp.step()
    if c4.button("Stop"):
        st.session_state.simulation_status = "stopped"
    c5, c6 = st.columns(2)
    if c5.button("Process next chunk"):
        if exp is None:
            st.warning("Train an ensemble first.")
        elif st.session_state.simulation_status in ("paused", "stopped"):
            st.warning("Resume or start the simulation before processing another chunk.")
        elif not exp.complete:
            exp.step()
            st.session_state.simulation_status = "running"
    if c6.button("Run to completion"):
        if exp is None:
            st.warning("Train an ensemble first.")
        else:
            with st.spinner("Evaluating the remaining ordered stream..."):
                exp.run_to_completion()
            st.session_state.simulation_status = "complete"
            st.success("Stream evaluation completed.")
    if st.button("Reset simulation"):
        try:
            st.session_state.experiment = new_experiment()
            st.session_state.simulation_status = "ready"
            st.success("Experiment reset and ensemble retrained on the initial training segment.")
        except Exception as exc:
            st.error(f"Could not reset: {exc}")
    exp = st.session_state.experiment
    if exp:
        metrics_cards(exp)
        st.progress(min(1.0, exp.position / len(exp.X)), text=f"Stream position {exp.position:,} / {len(exp.X):,}")
        st.caption(f"Simulation state: {st.session_state.simulation_status} · next chunk: {exp.chunk_number + 1}")
        if not exp.results_frame().empty:
            st.plotly_chart(accuracy_chart(exp.results_frame()), use_container_width=True)
            st.plotly_chart(member_accuracy_chart(exp.results_frame()), use_container_width=True)


def page_drift():
    st.title("Drift Detection")
    exp = active_experiment()
    if not exp:
        return
    st.write(f"Ensemble Page-Hinkley monitor · delta={exp.delta:g} · threshold={exp.threshold:g}")
    if exp.monitor_records:
        records = exp.monitors_frame()
        st.plotly_chart(monitor_chart(records), use_container_width=True)
        st.plotly_chart(reconstruction_timeline(exp.events), use_container_width=True)
        latest = records.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current statistic", f"{latest['ensemble_statistic']:.4f}")
        c2.metric("Running mean error", f"{latest['ensemble_mean_error']:.3f}")
        c3.metric("Threshold", f"{exp.threshold:.3f}")
        st.dataframe(records.tail(100), use_container_width=True)
    else:
        st.info("Monitor statistics will appear after at least one stream chunk is evaluated.")
    st.subheader("Alarm and activity history")
    st.dataframe(pd.DataFrame(exp.events), use_container_width=True)


def page_reconstruction():
    st.title("Model Reconstruction")
    exp = active_experiment()
    if not exp:
        return
    st.write("Reconstruction is executed only after a Page-Hinkley alarm. Candidate validation uses the latest window in temporal order.")
    cols = st.columns(len(exp.model.members))
    for i, (column, member) in enumerate(zip(cols, exp.model.members)):
        with column:
            st.markdown(f"**Member {i + 1}**")
            st.info(member.name)
            st.caption(f"Weight {member.weight:.3f} · smoothed error {member.smoothed_error:.3f}")
    if exp.reconstruction_log:
        st.subheader("Reconstruction log")
        st.dataframe(exp.reconstruction_frame(), use_container_width=True)
        st.plotly_chart(reconstruction_comparison_chart(exp.reconstruction_frame()), use_container_width=True)
        latest = exp.reconstruction_log[-1]
        st.subheader(f"Latest reconstruction · member {latest['member']}")
        st.write(f"{latest['previous_learner']} → **{latest['new_learner']}**")
        st.caption(f"Reason: {latest['reason']} · window: {latest['window_size']} · "
                   f"validation accuracy: {latest['validation_performance']:.1%} · "
                   f"time: {latest['duration_seconds']:.3f}s")
        st.dataframe(pd.DataFrame(latest["candidate_results"]), use_container_width=True)
    else:
        st.info("No actual reconstruction has occurred in this run.")


def page_performance():
    st.title("Model Performance")
    exp = active_experiment()
    if not exp:
        return
    if exp.processed_labels:
        metrics = classification_metrics(exp.processed_labels, exp.processed_predictions)
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
        cols[1].metric("Macro precision", f"{metrics['precision_macro']:.2%}")
        cols[2].metric("Macro recall", f"{metrics['recall_macro']:.2%}")
        cols[3].metric("Macro F1", f"{metrics['f1_macro']:.2%}")
        st.write("Confusion matrix (rows = actual classes, columns = predicted classes):")
        st.dataframe(pd.DataFrame(
            metrics["confusion_matrix"], index=metrics["classes"], columns=metrics["classes"]
        ), use_container_width=False)
        if metrics["false_positive_rate"] is not None:
            st.metric("False positive rate", f"{metrics['false_positive_rate']:.2%}")
        if not exp.results_frame().empty:
            st.plotly_chart(accuracy_chart(exp.results_frame()), use_container_width=True)
            st.plotly_chart(member_accuracy_chart(exp.results_frame()), use_container_width=True)
        st.plotly_chart(weight_chart(exp.model.members), use_container_width=True)
        if not exp.results_frame().empty:
            st.plotly_chart(weight_history_chart(exp.results_frame()), use_container_width=True)
        if exp.reconstruction_log:
            reconstruction_frame = exp.reconstruction_frame()
            st.plotly_chart(reconstruction_comparison_chart(reconstruction_frame), use_container_width=True)
            reconstruction_frame["reconstruction_delay_instances"] = 0
            st.caption("Alarm-to-reconstruction delay is zero for this synchronous chunk processor.")
            st.plotly_chart(accuracy_cost_chart(reconstruction_frame), use_container_width=True)
            st.plotly_chart(px.bar(
                reconstruction_frame, x="reconstruction_id",
                y="reconstruction_delay_instances", title="Alarm-to-reconstruction delay",
                template="plotly_dark",
            ), use_container_width=True)
        st.dataframe(pd.DataFrame([
            {"Member": i + 1, "Learner": member.name, "Weight": member.weight,
             "Smoothed error": member.smoothed_error}
            for i, member in enumerate(exp.model.members)
        ]), use_container_width=True)
    else:
        st.info("No predictions have been evaluated yet.")


def page_comparison():
    st.title("Algorithm Comparison")
    st.write("All methods use the same ordered dataset and initial training segment. DWM-style is a simplified research baseline, not the complete published DWM algorithm.")
    if st.button("Run comparison", type="primary"):
        try:
            X, y, _ = selected_data()
            cfg = st.session_state.config
            with st.spinner("Training and evaluating five methods on the same stream..."):
                summary, outcomes = run_comparison(
                    X, y, chunk_size=int(cfg["chunk_size"]),
                    burn_in=int(cfg["burn_in"]), window_size=int(cfg["window_size"]),
                    random_seed=int(cfg["random_seed"]),
                )
            st.session_state.comparison = summary
            st.session_state.comparison_outcomes = outcomes
            st.success("Comparison completed using actual model predictions.")
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
    if st.session_state.comparison is not None:
        st.dataframe(st.session_state.comparison, use_container_width=True)
        st.plotly_chart(comparison_chart(st.session_state.comparison), use_container_width=True)
        st.plotly_chart(reconstruction_count_chart(st.session_state.comparison), use_container_width=True)
        outcomes = st.session_state.comparison_outcomes or {}
        curves = pd.DataFrame({
            name: pd.Series(result["accuracy_by_chunk"])
            for name, result in outcomes.items()
        })
        if not curves.empty:
            curves.index = np.arange(1, len(curves) + 1)
            curves.index.name = "Chunk"
            st.plotly_chart(px.line(curves, title="Accuracy per chunk", template="plotly_dark"), use_container_width=True)


def page_history():
    st.title("Experiment History")
    exp = st.session_state.experiment
    if exp:
        st.subheader("Save current experiment")
        name = st.text_input("Unique experiment name")
        if st.button("Save experiment"):
            try:
                summary = exp.summary()
                ExperimentDB().save(
                    name, exp.dataset_name, st.session_state.config,
                    summary, exp.reconstruction_log,
                )
                st.success("Experiment saved. Existing records are never overwritten.")
            except Exception as exc:
                st.error(f"Could not save experiment: {exc}")
    records = ExperimentDB().list()
    if records:
        rows = []
        for item in records:
            summary = json.loads(item["summary"])
            rows.append({
                "ID": item["id"], "Name": item["name"], "Created": item["created_at"],
                "Dataset": item["dataset"], "Accuracy": summary.get("accuracy"),
                "Macro F1": summary.get("f1_macro"),
                "Reconstructions": summary.get("reconstruction_count"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        selected_id = st.selectbox("Inspect saved experiment", [item["id"] for item in records])
        record = next(item for item in records if item["id"] == selected_id)
        st.json({
            "configuration": json.loads(record["configuration"]),
            "summary": json.loads(record["summary"]),
            "reconstruction_log": json.loads(record["reconstruction_log"]),
        })
    else:
        st.info("No saved experiments yet.")


def page_reports():
    st.title("Reports and Downloads")
    exp = st.session_state.experiment
    if exp:
        st.download_button(
            "Download prediction results CSV",
            exp.predictions_frame().to_csv(index=False).encode("utf-8"),
            "predictions.csv", "text/csv",
        )
        st.download_button(
            "Download reconstruction log CSV",
            exp.reconstruction_frame().to_csv(index=False).encode("utf-8"),
            "reconstruction_log.csv", "text/csv",
        )
        st.download_button(
            "Download performance metrics CSV",
            pd.DataFrame([exp.summary()]).to_csv(index=False).encode("utf-8"),
            "performance_metrics.csv", "text/csv",
        )
        st.download_button(
            "Download experiment summary PDF",
            build_pdf(exp), "see_pmr_experiment_summary.pdf", "application/pdf",
        )
        model_buffer = io.BytesIO()
        joblib.dump(exp.model, model_buffer)
        st.download_button(
            "Download trained ensemble",
            model_buffer.getvalue(), "see_pmr_ensemble.joblib", "application/octet-stream",
        )
    else:
        st.info("Run an experiment to enable its results and model downloads.")
    if st.session_state.preprocessed is not None:
        st.download_button(
            "Download preprocessed features CSV",
            st.session_state.preprocessed.to_csv(index=False).encode("utf-8"),
            "processed_features.csv", "text/csv",
        )
    if st.session_state.comparison is not None:
        st.download_button(
            "Download algorithm comparison CSV",
            st.session_state.comparison.to_csv(index=False).encode("utf-8"),
            "algorithm_comparison.csv", "text/csv",
        )


def page_overview():
    st.title("Self-Evolving Ensemble Learning System")
    st.caption("Performance-Triggered Intelligent Model Reconstruction")
    st.markdown("Monitor heterogeneous models against an ordered stream, detect sustained error changes, and reconstruct affected learners from recent observations.")
    exp = st.session_state.experiment
    if exp:
        metrics_cards(exp)
        left, right = st.columns([1.5, 1])
        if not exp.results_frame().empty:
            left.plotly_chart(accuracy_chart(exp.results_frame()), use_container_width=True)
        right.plotly_chart(weight_chart(exp.model.members), use_container_width=True)
        st.subheader("System activity")
        st.dataframe(pd.DataFrame(exp.events[-20:]), use_container_width=True)
    else:
        st.info("Load a dataset, then train the ensemble to begin an experiment. Dashboard metrics and charts appear only after real model execution.")
        st.subheader("Included learners")
        st.write("Decision tree (depth 3) · Decision tree (depth 6) · Gaussian Naive Bayes · Logistic Regression · KNN (up to 7 neighbors)")


def page_about():
    st.title("About SEE-PMR")
    st.write(
        "SEE-PMR is a research-oriented demonstration of a heterogeneous classifier ensemble "
        "with prequential evaluation, Page-Hinkley error monitoring, and validation-guided "
        "model replacement. The stream is evaluated in source order: each chunk is predicted "
        "before its labels are used for monitoring or reconstruction."
    )
    st.markdown("""
    **Important interpretation notes**
    - Uploaded files are static local datasets; they are not live production feeds.
    - Synthetic drift is deliberately introduced by a generator and is labeled as such.
    - Reconstruction requires an observed Page-Hinkley alarm and a usable recent window.
    - The DWM-style comparison is a simplified baseline.
    - Validation is temporal: the earlier 70% of the recent window is used to train candidates, and the later 30% to select one.
    """)
    st.code("python -m venv venv\nvenv\\Scripts\\activate\npip install -r requirements.txt\nstreamlit run app.py", language="powershell")


def main():
    init_state()
    page = sidebar()
    pages = {
        "Overview Dashboard": page_overview,
        "Dataset Management": page_dataset,
        "Data Preprocessing": page_preprocessing,
        "Ensemble Configuration": page_configuration,
        "Model Training": page_training,
        "Live Stream Simulation": page_simulation,
        "Drift Detection": page_drift,
        "Model Reconstruction": page_reconstruction,
        "Model Performance": page_performance,
        "Algorithm Comparison": page_comparison,
        "Experiment History": page_history,
        "Reports and Downloads": page_reports,
        "About SEE-PMR": page_about,
    }
    pages[page]()


if __name__ == "__main__":
    main()
