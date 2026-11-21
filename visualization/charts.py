"""Reusable Plotly charts built exclusively from experiment output."""

import plotly.express as px
import plotly.graph_objects as go


THEME = "plotly_dark"


def accuracy_chart(records):
    fig = px.line(
        records, x="position", y="accuracy", markers=True,
        title="Prequential chunk accuracy",
        labels={"position": "Instances processed", "accuracy": "Accuracy"},
        template=THEME,
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def comparison_chart(summary):
    fig = px.bar(
        summary, x="Method", y="Mean prequential accuracy", color="Method",
        title="Accuracy comparison", template=THEME,
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def monitor_chart(records):
    fig = go.Figure()
    if "ensemble_statistic" in records:
        fig.add_trace(go.Scatter(
            x=records["position"], y=records["ensemble_statistic"],
            mode="lines", name="Ensemble PH statistic",
        ))
    fig.update_layout(
        title="Page-Hinkley statistic over stream position",
        xaxis_title="Instances processed", yaxis_title="Statistic",
        template=THEME,
    )
    return fig


def weight_chart(members):
    fig = px.bar(
        x=[member.name for member in members],
        y=[member.weight for member in members],
        labels={"x": "Ensemble member", "y": "Weight"},
        title="Current normalized member weights",
        template=THEME,
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def reconstruction_timeline(events):
    alarms = [event for event in events if "alarm" in event["event"].lower()]
    if not alarms:
        return go.Figure().update_layout(
            title="Observed Page-Hinkley alarms",
            xaxis_title="Instances processed",
            template=THEME,
        )
    fig = px.scatter(
        alarms, x="position", y=[1] * len(alarms), color="event",
        title="Observed Page-Hinkley alarms",
        labels={"position": "Instances processed", "y": ""},
        template=THEME,
    )
    return fig


def member_accuracy_chart(records):
    columns = [column for column in records.columns if column.startswith("member_") and column.endswith("_accuracy")]
    fig = px.line(
        records, x="position", y=columns, markers=True,
        title="Member accuracy by chunk", template=THEME,
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def weight_history_chart(records):
    columns = [column for column in records.columns if column.startswith("member_") and column.endswith("_weight")]
    fig = px.line(
        records, x="position", y=columns, markers=True,
        title="Ensemble weight changes", template=THEME,
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def reconstruction_comparison_chart(reconstructions):
    if reconstructions.empty:
        return go.Figure().update_layout(title="No reconstruction events", template=THEME)
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=reconstructions["reconstruction_id"], y=reconstructions["previous_performance"],
        name="Before",
    ))
    figure.add_trace(go.Bar(
        x=reconstructions["reconstruction_id"], y=reconstructions["validation_performance"],
        name="Selected validation",
    ))
    figure.update_layout(
        barmode="group", title="Performance before vs selected validation",
        xaxis_title="Reconstruction", yaxis_title="Accuracy", template=THEME,
    )
    figure.update_yaxes(range=[0, 1])
    return figure


def reconstruction_count_chart(summary):
    return px.bar(
        summary, x="Method", y="Reconstructed members", color="Method",
        title="Reconstructed members by method", template=THEME,
    )


def accuracy_cost_chart(reconstructions):
    return px.scatter(
        reconstructions, x="duration_seconds", y="validation_performance",
        size="window_size", color="new_learner",
        hover_data=["member", "reason", "position"],
        labels={"duration_seconds": "Reconstruction time (seconds)",
                "validation_performance": "Validation accuracy"},
        title="Reconstruction accuracy vs computational cost",
        template=THEME,
    )


