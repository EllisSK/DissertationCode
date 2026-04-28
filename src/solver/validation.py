# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd

from pathlib import Path
from .flume import Flume

def reproduce_friction_experiments():
    path = Path("data/ManningsNExperiments.csv")
    df = pd.read_csv(path)

    
    target_x_mm = [2500, 5000, 7500]
    
    dx = 0.1
    length = 12.5
    N = int(length / dx)
    x_vals = np.linspace(dx / 2, length - (dx / 2), N)
    
    grouped = df.groupby(["Set Flow (l/s)", "Incline (%)"])
    
    for (flow_ls, incline_pct), group in grouped:
        print(f"\n--- Flow: {flow_ls} l/s | Incline: {incline_pct}% ---")
        
        exp_depths = []
        for x_loc in target_x_mm:
            avg_depth = group[group["X Position (mm)"] == x_loc]["Depth (mm)"].mean()
            exp_depths.append(avg_depth)
            
        print(f"Experimental Average Depths (mm) at X={target_x_mm}:")
        print(f"  {[round(d, 2) for d in exp_depths]}")
        
        flow_m3s = flow_ls / 1000.0
        incline_fraction = incline_pct / 100.0
        
        try:
            flume = Flume(barrier_setup=None, set_flow=flow_m3s, incline=incline_fraction)
            profile = flume.simulate()
            
            eta = profile[1:-1, 0]
            
            zb = -incline_fraction * x_vals
            
            depth_m = np.maximum(eta - zb, 0.0)
            depth_mm = depth_m * 1000.0
            
            sim_depths = []
            for x_loc in target_x_mm:
                x_loc_m = x_loc / 1000.0
                depth_at_x = np.interp(x_loc_m, x_vals, depth_mm)
                sim_depths.append(depth_at_x)
                
            print(f"Solver Predicted Depths (mm) at X={target_x_mm}:")
            print(f"  {[round(d, 2) for d in sim_depths]}")
            
        except Exception as e:
            print(f"  Solver failed to run. Ensure that the correct paths (like frictionValues.csv) are accessible. Error: {e}")

def reproduce_barrier_experiments():
    path = Path("data/BarrierExperiments.csv")
    df = pd.read_csv(path)
    
    dx = 0.1
    length = 12.5
    N = int(length / dx)
    x_vals = np.linspace(dx / 2, length - (dx / 2), N)
    
    grouped = df.groupby(["Barrier Setup", "Set Flow (l/s)"])
    
    incline_fraction = 0.0
    
    for (barrier_setup, flow_ls), group in grouped:
        print(f"\n--- Barrier Setup: {barrier_setup} | Flow: {flow_ls} l/s ---")
        
        target_x_mm = sorted(group["X Position (mm)"].unique())
        
        exp_depths = []
        for x_loc in target_x_mm:
            avg_depth = group[group["X Position (mm)"] == x_loc]["Depth (mm)"].mean()
            exp_depths.append(avg_depth)
            
        print(f"Experimental Average Depths (mm) at X={target_x_mm}:")
        print(f"  {[round(d, 2) for d in exp_depths]}")
        
        flow_m3s = flow_ls / 1000.0
        
        try:
            flume = Flume(barrier_setup=barrier_setup, set_flow=flow_m3s, incline=incline_fraction)
            profile = flume.simulate()
            
            eta = profile[1:-1, 0]
            
            zb = -incline_fraction * x_vals
            depth_m = np.maximum(eta - zb, 0.0)
            depth_mm = depth_m * 1000.0
            
            sim_depths = []
            for x_loc in target_x_mm:
                x_loc_m = x_loc / 1000.0
                depth_at_x = np.interp(x_loc_m, x_vals, depth_mm)
                sim_depths.append(depth_at_x)
                
            print(f"Solver Predicted Depths (mm) at X={target_x_mm}:")
            print(f"  {[round(d, 2) for d in sim_depths]}")
            
        except Exception as e:
            print(f"  Solver failed to run. Ensure that the correct paths (like frictionValues.csv) are accessible. Error: {e}")