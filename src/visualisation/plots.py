# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from .baseplots import *
from .io import *
from src.analysis import *


def visualisation_1_1(lab_data: pd.DataFrame):
    simple_combined_model = SimpleCombinedModel("simpleCombined", lab_data)

    setups = ["100-100-50", "100-0-0", "0-0-0"]

    us_depth_fig = create_flow_us_depth_plot(simple_combined_model.df, setups, title="Simple Combined Model Performance for 3 Barrier Setups")
    for setup in setups:
        min_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].min()/1000
        max_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].max()/1000
        add_function_to_plot(us_depth_fig, simple_combined_model.plotting_function, (min_us_depth, max_us_depth), 0.001, f"{setup} Model", 1000, 1, setup)
    save_figure(us_depth_fig, "Figure1_1", "Methodology")

    return us_depth_fig