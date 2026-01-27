# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from typing import Callable


def create_flow_head_plot(df: pd.DataFrame) -> go.Figure:
    df_sorted = df.sort_values(by="Total Head (m)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x= df_sorted[df_sorted["Barrier Setup"] == "50-0-0"]["Total Head (m)"],
        y= df_sorted[df_sorted["Barrier Setup"] == "50-0-0"]["Flow (m3/s)"],
        name="50",
        mode="lines+markers",
        marker_symbol="x"
    ))

    fig.add_trace(go.Scatter(
        x= df_sorted[df_sorted["Barrier Setup"] == "100-0-0"]["Total Head (m)"],
        y= df_sorted[df_sorted["Barrier Setup"] == "100-0-0"]["Flow (m3/s)"],
        name="100",
        mode="lines+markers",
        marker_symbol="x"
    ))

    fig.add_trace(go.Scatter(
        x= df_sorted[df_sorted["Barrier Setup"] == "150-0-0"]["Total Head (m)"],
        y= df_sorted[df_sorted["Barrier Setup"] == "150-0-0"]["Flow (m3/s)"],
        name="150",
        mode="lines+markers",
        marker_symbol="x"
    ))

    fig.update_layout(
        title="Sluice Gate Flow for Different Gap Sizes",
        xaxis={
            "title" : "Total Head (m)",
            "range" : [0, None]
        },
        yaxis={
            "title" : "Flow (m3/s)",
            "range" : [0, None]
        },
        legend_title_text="Gap Size (mm)",
    )

    return fig


def create_flow_depth_plot(df: pd.DataFrame) -> go.Figure:
    df_sorted = df.sort_values(by="Mean Upstream Depth (mm)")

    fig = px.scatter(
        df_sorted,
        x="Mean Upstream Depth (mm)",
        y="Set Flow (l/s)",
        color="Barrier Setup",
        title=f"Flow against upstream depth for multiple barrier configurations",
        template="presentation",
    )

    fig.update_traces(mode="lines+markers", marker_symbol="x")

    return fig


def create_sluice_cd_plot(df: pd.DataFrame) -> go.Figure:
    df_sorted = df.sort_values(by="Head (m)")

    fig = px.scatter(
        df_sorted,
        x="Head (m)",
        y="Coefficient of Discharge",
        color="Sluice Area (m2)",
        title=f"Cd against head for mutliple sluice areas (also bottom gap)",
        template="presentation",
    )

    fig.update_traces(marker_symbol="x")

    return fig


def create_sluice_cd_ha_plot(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="H/a",
        y="Coefficient of Discharge",
        color="Sluice Area (m2)",
        title=f"Cd against head/a for mutliple sluice areas (a: opening size)",
        template="presentation",
    )

    fig.update_traces(marker_symbol="x")

    return fig


def add_function_to_plot(
    fig: go.Figure,
    function: Callable,
    x_range: tuple,
    resolution: float,
    func_name: str,
) -> go.Figure:
    n = int((x_range[1] - x_range[0]) / resolution)
    x_array = np.linspace(x_range[0], x_range[1], n)
    y_array = np.empty(len(x_array))

    for i in range(len(x_array)):
        x = x_array[i]
        y = function(x)
        y_array[i] = y

    fig.add_trace(go.Scatter(x=x_array, y=y_array, mode="lines", name=func_name))

    return fig
