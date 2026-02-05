# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from typing import Callable


def create_flow_us_depth_plot(df: pd.DataFrame, setups: list[str], title: str = "Barrier Flow for Different Geometric Configurations") -> go.Figure:
    df_sorted = df.sort_values(by="Mean Upstream Depth (mm)")

    fig = go.Figure()

    for setup in setups:
        fig.add_trace(go.Scatter(
            x= df_sorted[df_sorted["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"],
            y= df_sorted[df_sorted["Barrier Setup"] == setup]["Flow (m3/s)"],
            name=setup,
            mode="lines+markers",
            marker_symbol="x"
        ))

    fig.update_layout(
        title=title,
        xaxis={
            "title" : "Mean Upstream Depth (mm)",
            "range" : [0, None]
        },
        yaxis={
            "title" : "Flow (m3/s)",
            "range" : [0, None]
        },
        legend_title_text="Barrier Setup (mm)",
    )

    return fig


def add_function_to_plot(
    fig: go.Figure,
    function: Callable,
    x_range: tuple,
    resolution: float,
    func_name: str,
    x_multiplier: int = 1,
    y_multiplier: int = 1,
    *args
) -> go.Figure:
    n = int((x_range[1] - x_range[0]) / resolution)
    x_array = np.linspace(x_range[0], x_range[1], n)
    y_array = np.empty(len(x_array))

    for i in range(len(x_array)):
        x = x_array[i]
        y = function(x, *args)
        y_array[i] = y

    x_array = x_array * x_multiplier
    y_array = y_array * y_multiplier

    fig.add_trace(go.Scatter(x=x_array, y=y_array, mode="lines", name=func_name))

    return fig
