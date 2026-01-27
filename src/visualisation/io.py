# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import os
from pathlib import Path
import plotly.graph_objects as go


def save_figure(fig: go.Figure, name: str, sub_directory: str = "misc"):
    figures_directory = Path("exports/figures")
    figures_directory.mkdir(parents=True, exist_ok=True)

    save_directory = figures_directory / sub_directory
    save_directory.mkdir(parents=True, exist_ok=True)

    fig.write_image(save_directory / f"{name}.svg", width=1592, height=984)
