# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import re

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from .baseplots import *
from .io import *
from src.analysis import *
from tqdm import tqdm


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

def visualisation_1_2(lab_data: pd.DataFrame):
    simple_combined_model = SimpleIFCombinedModel("simpleCombined", lab_data)
    simple_combined_model.fit()

    setups = ["100-100-50", "100-0-0", "0-0-0"]

    us_depth_fig = create_flow_us_depth_plot(simple_combined_model.df, setups, title="Simple IF Model Performance for 3 Barrier Setups")
    for setup in setups:
        min_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].min()/1000
        max_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].max()/1000
        add_function_to_plot(us_depth_fig, simple_combined_model.plotting_function, (min_us_depth, max_us_depth), 0.001, f"{setup} Model", 1000, 1, setup)
    save_figure(us_depth_fig, "Figure1_2", "Methodology")

    return us_depth_fig

def visualisation_1_3(lab_data: pd.DataFrame):
    simple_combined_model = SimpleCombinedModel("simpleCombined", lab_data)

    setups = simple_combined_model.df["Barrier Setup"].unique()

    print("Creating plots for the simple combined model:")
    for setup in tqdm(setups):
        us_depth_fig = create_flow_us_depth_plot(simple_combined_model.df, [setup], title=f"Simple Combined Model Performance for {setup}")
        min_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].min()/1000
        max_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].max()/1000
        add_function_to_plot(us_depth_fig, simple_combined_model.plotting_function, (min_us_depth, max_us_depth), 0.001, f"{setup} Model", 1000, 1, setup)
        save_figure(us_depth_fig, setup, "SimpleCombinedAllSetups")

def visualisation_1_4():
    us_profile = np.linspace(280, 280, num=5000)
    ds_profile = np.concatenate([np.full(1000, 61), 120.5 - 59.5 * np.cos(np.linspace(0, np.pi, 500)), np.full(3500, 180)])
    fig = create_barrier_depth_diagram("100-100-100", us_profile, ds_profile, "Test Plot - Sluice Flow Example")
    fig.savefig("exports/figures/barriertest.svg")

def visualisation_1_5(lab_data: pd.DataFrame):
    simple_combined_model = AdvancedCombinedModel("advancedCombined", lab_data)

    setups = simple_combined_model.df["Barrier Setup"].unique()

    print("Creating plots for the advanced combined model:")
    for setup in tqdm(setups):
        us_depth_fig = create_flow_us_depth_plot(simple_combined_model.df, [setup], title=f"Advanced Combined Model Performance for {setup}")
        min_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].min()/1000
        max_us_depth = simple_combined_model.df[simple_combined_model.df["Barrier Setup"] == setup]["Mean Upstream Depth (mm)"].max()/1000
        add_function_to_plot(us_depth_fig, simple_combined_model.plotting_function, (min_us_depth, max_us_depth), 0.001, f"{setup} Model", 1000, 1, setup)
        save_figure(us_depth_fig, setup, "AdvancedCombinedAllSetups")

def visualisation_1_6(lab_data: pd.DataFrame):
    data_directory = Path("exports/numerical/barriers")
    output_directory = Path("exports/figures/numerical/barriers/")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    file_paths = list(data_directory.glob("*-*-*-*.csv"))
    
    for file_path in tqdm(file_paths):
        parts = file_path.stem.split("-")
        barrier_setup = f"{parts[0]}-{parts[1]}-{parts[2]}"
        flow_rate = float(parts[3])
        
        df = pd.read_csv(file_path)
        x_mm = df["X Position (m)"].values * 1000
        depth = df["Depth (mm)"].values
        
        us_x = np.linspace(0, 5000, num=5000)
        ds_x = np.linspace(5000, 12500, num=7500)
        
        us_profile = np.interp(us_x, x_mm, depth)
        ds_profile = np.interp(ds_x, x_mm, depth)
        
        title = f"Barrier Setup: {barrier_setup} | Flow: {flow_rate} l/s"
        
        point_data = lab_data[
            (lab_data["Barrier Setup"] == barrier_setup) & 
            (lab_data["Set Flow (l/s)"] == flow_rate)
        ]
        
        if not point_data.empty and "X Position (mm)" in point_data.columns and "Depth (mm)" in point_data.columns:
            point_data = point_data.groupby("X Position (mm)", as_index=False)["Depth (mm)"].mean()
        
        fig = create_barrier_depth_diagram(barrier_setup, us_profile, ds_profile, point_data, title)
        fig.savefig(output_directory / f"{file_path.stem}.svg")
        plt.close(fig)

def visualisation_1_7(measured_friction_data: pd.DataFrame):
    data_directory = Path("exports/numerical/friction")
    output_directory = Path("exports/figures/numerical/friction/")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    for p in data_directory.glob("*.csv"):
        if re.match(r"^\d+-\d+(?:\.\d+)?\.csv$", p.name):
            file_paths.append(p)
            
    for file_path in tqdm(file_paths):
        parts = file_path.stem.split("-")
        incline_pct = float(parts[0]) / 10.0
        flow_rate = float(parts[1])
        
        df = pd.read_csv(file_path)
        x_mm = df["X Position (m)"].values * 1000
        depth = df["Depth (mm)"].values
        
        x_profile = np.linspace(0, 12500, num=12500)
        depth_profile = np.interp(x_profile, x_mm, depth)
        
        title = f"Friction Experiment | Incline: {incline_pct}% | Flow: {flow_rate} l/s"
        
        point_data = measured_friction_data[
            (measured_friction_data["Incline (%)"] == incline_pct) & 
            (measured_friction_data["Set Flow (l/s)"] == flow_rate)
        ]
        
        if not point_data.empty and "X Position (mm)" in point_data.columns and "Depth (mm)" in point_data.columns:
            point_data = point_data.groupby("X Position (mm)", as_index=False)["Depth (mm)"].mean()
        
        fig = create_friction_depth_diagram(incline_pct, x_profile, depth_profile, point_data, title)
        fig.savefig(output_directory / f"{file_path.stem}.svg")
        plt.close(fig)

def visualisation_1_8() -> go.Figure:
    df = pd.read_csv(Path("data/ManningsNExperiments.csv"))
    data = df.groupby(["Set Flow (l/s)", "Incline (%)", "X Position (mm)"])["Depth (mm)"].mean().reset_index()

    data["AOD (m)"] = ((10000 - data["X Position (mm)"]) / 1000) * (data["Incline (%)"] / 100)
    data["Velocity Head (m)"] = np.power((data["Set Flow (l/s)"] / data["Depth (mm)"]), 2) / (2 * 9.81)
    data["Depth (m)"] = data["Depth (mm)"] / 1000
    data["Total Head (m)"] = data["AOD (m)"] + data["Velocity Head (m)"] + data["Depth (m)"]
    data["X Position (m)"] = data["X Position (mm)"] / 1000

    def get_head_slope(group):
        slope, _ = np.polyfit(group["X Position (m)"], group["Total Head (m)"], 1)
        return slope
    
    slope_data = data.groupby(["Set Flow (l/s)", "Incline (%)"]).apply(get_head_slope).reset_index(name="Free Surface Slope")
    data = data.merge(slope_data, on=["Set Flow (l/s)", "Incline (%)"], how="left")
    
    data["Sf"] = -data["Free Surface Slope"]
    data["P (m)"] = 1 + (2 * data["Depth (m)"])
    data["Composite n"] = np.sqrt((data["Sf"] * np.power(data["Depth (m)"], 10/3)) / 
                                  (np.power(data["Set Flow (l/s)"] / 1000, 2) * np.power(data["P (m)"], 4/3)))

    x_vals = data["Depth (m)"]
    y_vals = (data["P (m)"] * np.power(data["Composite n"], 1.5)) / 2

    valid_idx = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    x_vals = x_vals[valid_idx]
    y_vals = y_vals[valid_idx]

    slope, intercept = np.polyfit(x_vals, y_vals, 1)

    ci_df = pd.read_csv(Path("exports/reports/frictionCIValues.csv"))
    lower_bed = ci_df.loc[ci_df["Bound"] == "Lower", "Bed"].values[0]
    upper_bed = ci_df.loc[ci_df["Bound"] == "Upper", "Bed"].values[0]
    lower_wall = ci_df.loc[ci_df["Bound"] == "Lower", "Wall"].values[0]
    upper_wall = ci_df.loc[ci_df["Bound"] == "Upper", "Wall"].values[0]

    x_line = np.linspace(0, x_vals.max() * 1.1, 100)
    y_line_nominal = slope * x_line + intercept
    y_line_lower = (lower_wall ** 1.5) * x_line + (lower_bed ** 1.5) / 2
    y_line_upper = (upper_wall ** 1.5) * x_line + (upper_bed ** 1.5) / 2

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([y_line_upper, y_line_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(0, 141, 255, 0.2)",
        line={"color": "rgba(255,255,255,0)"},
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval"
    ))

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line_nominal,
        mode="lines",
        line={"color": "#d83034", "width": 3, "dash": "dash"},
        name="Regression Fit"
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker={"symbol": "x", "color": "#008dff"},
        name="Variables Derived from Measurements"
    ))

    fig.update_layout(
        title="Friction Regression Fit with 95% CI",
        xaxis={
            "title": "Regression Variable X: Depth (m)",
            "range": [0, x_vals.max() * 1.1]
        },
        yaxis={
            "title": "Regression Variable Y",
            "range": [0, y_vals.max() * 1.1],
            "exponentformat": "power",
            "tickfont": {"size": 24}
        },
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
            "font": {"size": 32}
        }
    )

    save_figure(fig, "FrictionFit")
    return fig