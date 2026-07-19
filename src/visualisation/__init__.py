# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from .baseplots import (
    add_function_to_plot,
    create_barrier_depth_diagram,
    create_flow_us_depth_plot,
    create_friction_depth_diagram,
)
from .io import save_figure
from .paper import generate_paper_figures
from .plots import (
    visualisation_1_1,
    visualisation_1_2,
    visualisation_1_3,
    visualisation_1_5,
    visualisation_1_6,
    visualisation_1_7,
    visualisation_1_8,
)
from .template import create_custom_template

create_custom_template()

__all__ = [
    "add_function_to_plot",
    "create_barrier_depth_diagram",
    "create_custom_template",
    "create_flow_us_depth_plot",
    "create_friction_depth_diagram",
    "generate_paper_figures",
    "save_figure",
    "visualisation_1_1",
    "visualisation_1_2",
    "visualisation_1_3",
    "visualisation_1_5",
    "visualisation_1_6",
    "visualisation_1_7",
    "visualisation_1_8",
]
