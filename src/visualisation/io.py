# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path

import plotly.graph_objects as go


def save_figure(
    fig: go.Figure,
    name: str,
    sub_directory: str = "misc",
    half: bool = False,
    formats: tuple[str, ...] = ("svg",),
):
    figures_directory = Path("exports/figures")
    figures_directory.mkdir(parents=True, exist_ok=True)

    save_directory = figures_directory / sub_directory
    save_directory.mkdir(parents=True, exist_ok=True)

    stem = name if not half else f"{name}_half"
    height = 984 if not half else 984 / 2

    for image_format in formats:
        fig.write_image(
            save_directory / f"{stem}.{image_format}", width=1592, height=height
        )
