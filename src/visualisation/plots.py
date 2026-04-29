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