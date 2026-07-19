# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Publication figures for the research paper (Journal of Flood Risk Management).

Each function produces one numbered figure from the manuscript and writes it to
exports/figures/paper/. Figure 1 (CAD drawing of the physical barrier model) and
Figure 4 (photograph of the flume) are produced outside this module.

Colour assignments avoid red-green combinations per the journal's accessibility
guidance, and every multi-series figure additionally distinguishes series by
line style or marker shape so identity never relies on colour alone.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

from src.analysis import (
    AdvancedCombinedModel,
    AdvancedSluiceModel,
    AdvancedWeirModel,
    SimpleCombinedModel,
    read_barrier_data,
    remove_submerged,
)
from src.constants import (
    CHANNEL_WIDTH,
    FLUME_LENGTH,
    PLANK_1_HEIGHT,
    PLANK_2_HEIGHT,
    PLANK_3_HEIGHT,
)
from src.solver import Flume

from .baseplots import create_barrier_depth_diagram
from .io import save_figure
from .plots import visualisation_1_8

MEASURED_COLOUR = "#008dff"
PRIMARY_MODEL_COLOUR = "#d83034"
SECONDARY_MODEL_COLOUR = "#ff9d3a"
TERTIARY_MODEL_COLOUR = "#c701ff"

PAPER_SUBDIR = "paper"
PAPER_FIGURE_DIR = Path("exports/figures") / PAPER_SUBDIR

FLOW_AXIS_TITLE = "Flow (m<sup>3</sup> s<sup>−1</sup>)"
DEPTH_AXIS_TITLE = "Mean Upstream Depth (mm)"


def _measured_trace(model_df: pd.DataFrame, setup: str, **overrides) -> go.Scatter:
    data = model_df[model_df["Barrier Setup"] == setup].sort_values("Mean Upstream Depth (mm)")
    trace = {
        "x": data["Flow (m3/s)"],
        "y": data["Mean Upstream Depth (mm)"],
        "mode": "markers",
        "name": "Measured",
        "marker": {"symbol": "x", "color": MEASURED_COLOUR},
    }
    trace.update(overrides)
    return go.Scatter(**trace)


def _model_curve(plotting_function, setup: str, depth_range_m: tuple, name: str, colour: str, dash: str = "solid", **overrides) -> go.Scatter:
    depths = np.arange(depth_range_m[0], depth_range_m[1], 0.0005)
    flows = np.array([plotting_function(depth, setup) for depth in depths])
    trace = {
        "x": flows,
        "y": depths * 1000,
        "mode": "lines",
        "name": name,
        "line": {"color": colour, "dash": dash},
    }
    trace.update(overrides)
    return go.Scatter(**trace)


def _setup_depth_range(model_df: pd.DataFrame, setup: str) -> tuple:
    setup_depths = model_df[model_df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"]
    return setup_depths.min() / 1000, setup_depths.max() / 1000


def figure_2_combined_models(lab_data: pd.DataFrame):
    """Measured data and both combined models for 50-50-50, with an inset
    highlighting the non-monotonic operation mode transition of the simple model."""
    setup = "50-50-50"

    simple = SimpleCombinedModel("simpleCombined", lab_data)
    advanced = AdvancedCombinedModel("advancedCombined", lab_data)

    depth_range = _setup_depth_range(simple.df, setup)

    fig = go.Figure()
    fig.add_trace(_measured_trace(simple.df, setup))
    fig.add_trace(_model_curve(simple.plotting_function, setup, depth_range, "Simple combined model", PRIMARY_MODEL_COLOUR))
    fig.add_trace(_model_curve(advanced.plotting_function, setup, depth_range, "Advanced combined model", SECONDARY_MODEL_COLOUR, dash="dash"))

    # Inset window around the transition at the bottom of the second plank,
    # where the simple model switches its weir term to an orifice term
    transition_depth = 0.05 + PLANK_1_HEIGHT + 0.05
    inset_depths = (transition_depth - 0.012, transition_depth + 0.018)
    window_flows = [simple.plotting_function(d, setup) for d in np.arange(*inset_depths, 0.0005)]
    inset_flows = (min(window_flows) * 0.995, max(window_flows) * 1.005)

    fig.add_trace(_model_curve(simple.plotting_function, setup, inset_depths, "", PRIMARY_MODEL_COLOUR, showlegend=False, xaxis="x2", yaxis="y2"))
    fig.add_trace(_model_curve(advanced.plotting_function, setup, inset_depths, "", SECONDARY_MODEL_COLOUR, dash="dash", showlegend=False, xaxis="x2", yaxis="y2"))

    fig.add_shape(
        type="rect",
        x0=inset_flows[0], x1=inset_flows[1],
        y0=inset_depths[0] * 1000, y1=inset_depths[1] * 1000,
        line={"color": "black", "width": 2},
    )

    inset_axis_style = {
        "showline": True,
        "linecolor": "black",
        "mirror": True,
        "ticks": "outside",
        "tickfont": {"size": 22},
        "showgrid": False,
    }
    fig.update_layout(
        xaxis={"title": FLOW_AXIS_TITLE, "range": [0, None]},
        yaxis={"title": DEPTH_AXIS_TITLE, "range": [0, None]},
        xaxis2={"domain": [0.56, 0.97], "anchor": "y2", "range": list(inset_flows), **inset_axis_style},
        yaxis2={"domain": [0.07, 0.52], "anchor": "x2", "range": [d * 1000 for d in inset_depths], **inset_axis_style},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    save_figure(fig, "Figure2", PAPER_SUBDIR)
    return fig


def figure_3_advanced_model(lab_data: pd.DataFrame):
    """Measured data and the advanced combined model for the 0-0-100 configuration."""
    setup = "0-0-100"

    advanced = AdvancedCombinedModel("advancedCombined", lab_data)
    depth_range = _setup_depth_range(advanced.df, setup)

    fig = go.Figure()
    fig.add_trace(_measured_trace(advanced.df, setup))
    fig.add_trace(_model_curve(advanced.plotting_function, setup, depth_range, "Advanced combined model", PRIMARY_MODEL_COLOUR))

    fig.update_layout(
        xaxis={"title": FLOW_AXIS_TITLE, "range": [0, None]},
        yaxis={"title": DEPTH_AXIS_TITLE, "range": [0, None]},
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    save_figure(fig, "Figure3", PAPER_SUBDIR)
    return fig


def figure_5_representation_comparison(lab_data: pd.DataFrame):
    """Stage-discharge relationships for 50-50-50 under three barrier representations:
    a lumped friction coefficient, a weir-gate structure and the compound model.

    The lumped friction representation is a Manning-form rating
    Q = C h^(5/3) / (W + 2h)^(2/3) with the single coefficient C least-squares
    fitted to the measured data, mimicking a raised-roughness channel section.
    The weir-gate representation combines the advanced sluice equation for the
    bottom gap with the advanced weir equation over the top of the structure,
    omitting the orifice flow through the barrier body.
    """
    setup = "50-50-50"

    advanced = AdvancedCombinedModel("advancedCombined", lab_data)
    sluice = AdvancedSluiceModel("figure5Sluice", lab_data)
    weir = AdvancedWeirModel("figure5Weir", lab_data)

    gap1, gap2, gap3 = (int(g) / 1000 for g in setup.split("-"))
    crest = gap1 + PLANK_1_HEIGHT + gap2 + PLANK_2_HEIGHT + gap3 + PLANK_3_HEIGHT

    def weir_gate(depth, _setup):
        flow = sluice._equation((depth, gap1))
        if depth > crest:
            flow += weir._equation((depth - crest, crest))
        return flow

    setup_df = advanced.df[advanced.df["Barrier Setup"] == setup]

    def manning_form(h, coeff):
        return coeff * np.power(h, 5 / 3) / np.power(CHANNEL_WIDTH + 2 * h, 2 / 3)

    popt, _ = curve_fit(manning_form, setup_df["Upstream Head (m)"], setup_df["Flow (m3/s)"])

    def lumped_friction(depth, _setup):
        return manning_form(depth, popt[0])

    depth_range = _setup_depth_range(advanced.df, setup)

    fig = go.Figure()
    fig.add_trace(_measured_trace(advanced.df, setup))
    fig.add_trace(_model_curve(advanced.plotting_function, setup, depth_range, "Compound structure model", PRIMARY_MODEL_COLOUR))
    # Explicit dash pattern: the default scales with line width and swallows the
    # near-vertical section of the curve at the crest transition
    fig.add_trace(_model_curve(weir_gate, setup, depth_range, "Weir-gate representation", SECONDARY_MODEL_COLOUR, dash="14px,7px"))
    fig.add_trace(_model_curve(lumped_friction, setup, depth_range, "Lumped friction representation", TERTIARY_MODEL_COLOUR, dash="dot"))

    fig.update_layout(
        xaxis={"title": FLOW_AXIS_TITLE, "range": [0, None]},
        yaxis={"title": DEPTH_AXIS_TITLE, "range": [0, None]},
        # Bottom-right: the top-left corner would occlude the steep weir-gate curve
        legend={"yanchor": "bottom", "y": 0.03, "xanchor": "right", "x": 0.99},
    )

    save_figure(fig, "Figure5", PAPER_SUBDIR)
    return fig


def figure_6_friction_regression():
    """Composite Manning's n regression with its 95% confidence band."""
    fig = visualisation_1_8()
    fig.update_layout(title=None)
    save_figure(fig, "Figure6", PAPER_SUBDIR)
    return fig


FIGURE_7_CASES = [
    ("a", "100-50-0", 170.0),
    ("b", "0-0-0", 300.0),
    ("c", "50-0-0", 350.0),
    ("d", "50-50-50", 50.0),
]


def _ensure_numerical_profile(setup: str, flow_ls: float) -> Path:
    """Return the solver depth profile for a case, running the solver if absent."""
    path = Path("exports/numerical/barriers") / f"{setup}-{flow_ls}.csv"
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)

    flume = Flume(barrier_setup=setup, set_flow=flow_ls / 1000.0, incline=0.0)
    profile = flume.simulate()

    dx = 0.1
    n_cells = int(FLUME_LENGTH / dx)
    x_vals = np.linspace(dx / 2, FLUME_LENGTH - (dx / 2), n_cells)

    pd.DataFrame({
        "X Position (m)": x_vals,
        "Depth (mm)": np.maximum(profile[1:-1, 0], 0.0) * 1000.0,
        "Velocity (m/s)": profile[1:-1, 1],
    }).to_csv(path, index=False)

    return path


def figure_7_numerical_profiles(raw_lab_data: pd.DataFrame):
    """Four-panel comparison of numerical depth profiles against measurements.

    Expects the unfiltered measurement dataset: case (d) appears in the
    submerged-configuration list, and filtering it out would drop its measured
    points, including the supercritical jet measurement discussed in the paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(2 * 15.92 * 0.393701 / 1.6, 2 * 9.84 * 0.393701 / 1.6))

    for (label, setup, flow), ax in zip(FIGURE_7_CASES, axes.flat):
        profile_path = _ensure_numerical_profile(setup, flow)

        sim_df = pd.read_csv(profile_path)
        x_mm = sim_df["X Position (m)"].values * 1000
        depth = sim_df["Depth (mm)"].values

        us_profile = np.interp(np.linspace(0, 5000, num=5000), x_mm, depth)
        ds_profile = np.interp(np.linspace(5000, 12500, num=7500), x_mm, depth)

        point_data = raw_lab_data[
            (raw_lab_data["Barrier Setup"] == setup)
            & (raw_lab_data["Set Flow (l/s)"] == flow)
        ]
        point_data = point_data.groupby("X Position (mm)", as_index=False)["Depth (mm)"].mean()

        title = f"({label}) {setup} at {flow:.0f} L s$^{{-1}}$"
        create_barrier_depth_diagram(setup, us_profile, ds_profile, point_data, title, ax=ax)

        if label != "a":
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    fig.tight_layout()

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURE_DIR / "Figure7.svg")
    plt.close(fig)
    return fig


def generate_paper_figures():
    """Regenerate every code-produced figure in the manuscript."""
    lab_data = remove_submerged(read_barrier_data())
    raw_lab_data = pd.read_csv(Path("data/BarrierExperiments.csv"))

    figure_2_combined_models(lab_data)
    figure_3_advanced_model(lab_data)
    figure_5_representation_comparison(lab_data)
    figure_6_friction_regression()
    figure_7_numerical_profiles(raw_lab_data)
