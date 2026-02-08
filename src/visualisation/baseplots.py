# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator, FuncFormatter

from typing import Callable


def create_flow_us_depth_plot(df: pd.DataFrame, setups: list[str], title: str = "Barrier Flow for Different Geometric Configurations") -> go.Figure:
    df_sorted = df.sort_values(by="Mean Upstream Depth (mm)")

    fig = go.Figure()

    for setup in setups:
        fig.add_trace(go.Scatter(
            x= df_sorted[df_sorted["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"],
            y= df_sorted[df_sorted["Barrier Setup"] == setup]["Flow (m3/s)"],
            name=setup,
            mode="markers",
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

def create_barrier_depth_diagram(barrier_setup: str, us_profile: np.ndarray, ds_profile: np.ndarray, title: str = ""):
    FLUME_LENGTH = 10000
    FLUME_DEPTH = 800
    BARRIER_WIDTH = 15
    BARRIER_X_CENTER = 5000

    fig_width_in = 15.92 * 0.393701
    fig_height_in = (9.84 * 0.393701) / 2

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    water_colour = "aqua"

    us_x = np.linspace(0, 5000, num=5000)
    ax.plot(us_x, us_profile, color=water_colour)
    ax.fill_between(us_x, us_profile, 0, color=water_colour)

    ds_x = np.linspace(5000, 10000, num=5000)
    ax.plot(ds_x, ds_profile, color=water_colour)
    ax.fill_between(ds_x, ds_profile, 0, color=water_colour)

    gaps = list(map(int, barrier_setup.split("-")))
    current_y = gaps[0]

    planks_data = []
    planks_data.append((BARRIER_X_CENTER-BARRIER_WIDTH/2, current_y, BARRIER_WIDTH, 200))
    current_y += 200 + gaps[1]

    planks_data.append((BARRIER_X_CENTER-BARRIER_WIDTH/2, current_y, BARRIER_WIDTH, 100))
    current_y += 100 + gaps[2]

    planks_data.append((BARRIER_X_CENTER-BARRIER_WIDTH/2, current_y, BARRIER_WIDTH, 100))

    for (x, y, w, h) in planks_data:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="black", facecolor="black", zorder=10)
        ax.add_patch(rect)

    ax.set_xlim(0, FLUME_LENGTH)
    ax.set_ylim(0, FLUME_DEPTH)

    ax.yaxis.set_major_locator(MultipleLocator(100))
    
    ax.yaxis.set_minor_locator(MultipleLocator(25))

    def y_label_filter(x, pos):
        if x in [0, 400, 800]:
            return f"{int(x)}"
        return ""
    ax.yaxis.set_major_formatter(FuncFormatter(y_label_filter))

    ax.xaxis.set_major_locator(MultipleLocator(1000))

    ax.xaxis.set_minor_locator(MultipleLocator(100))

    def x_label_filter(x, pos):
        if x in [0, 5000, 10000]:
            return f"{int(x)}"
        return ""
    ax.xaxis.set_major_formatter(FuncFormatter(x_label_filter))

    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=3)

    plt.subplots_adjust(top=0.85, bottom=0.25)
    fig.text(0.02, 0.98, title, ha="left", va="top", fontweight="bold", fontsize=11)

    plt.xlabel("Position (mm)", fontname="Arial", fontdict={"size":11})
    plt.ylabel("Depth (mm)", fontname="Arial", fontdict={"size":11})

    return fig