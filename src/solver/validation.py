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
    
    dx = 0.1
    length = 12.5
    N = int(length / dx)
    x_vals = np.linspace(dx / 2, length - (dx / 2), N)
    
    grouped = df.groupby(["Set Flow (l/s)", "Incline (%)"])
    
    out_dir = Path("exports/numerical/friction")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for (flow_ls, incline_pct), group in grouped:
        flow_m3s = flow_ls / 1000.0
        incline_fraction = incline_pct / 100.0
        
        try:
            flume = Flume(barrier_setup=None, set_flow=flow_m3s, incline=incline_fraction)
            profile = flume.simulate()
            
            eta = profile[1:-1, 0]
            velocity = profile[1:-1, 1] 
            
            zb = -incline_fraction * x_vals
            
            depth_m = np.maximum(eta - zb, 0.0)
            depth_mm = depth_m * 1000.0
            
            results_df = pd.DataFrame({
                "X Position (m)": x_vals,
                "Depth (mm)": depth_mm,
                "Velocity (m/s)": velocity
            })
            
            filename = f"{int(incline_pct * 10)}-{int(flow_ls)}.csv"
            results_df.to_csv(out_dir / filename, index=False)
                
        except Exception as e:
            print(f"Solver failed to run for Flow: {flow_ls}, Incline: {incline_pct}. Error: {e}")

def reproduce_barrier_experiments():
    path = Path("data/BarrierExperiments.csv")
    df = pd.read_csv(path)
    
    dx = 0.1
    length = 12.5
    N = int(length / dx)
    x_vals = np.linspace(dx / 2, length - (dx / 2), N)
    
    grouped = df.groupby(["Barrier Setup", "Set Flow (l/s)"])
    
    incline_fraction = 0.0
    
    out_dir = Path("exports/numerical/barriers")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for (barrier_setup, flow_ls), group in grouped:
        flow_m3s = flow_ls / 1000.0
        
        try:
            flume = Flume(barrier_setup=barrier_setup, set_flow=flow_m3s, incline=incline_fraction)
            profile = flume.simulate()
            
            eta = profile[1:-1, 0]
            velocity = profile[1:-1, 1] 
            
            zb = -incline_fraction * x_vals
            depth_m = np.maximum(eta - zb, 0.0)
            depth_mm = depth_m * 1000.0
            
            results_df = pd.DataFrame({
                "X Position (m)": x_vals,
                "Depth (mm)": depth_mm,
                "Velocity (m/s)": velocity
            })
            
            filename = f"{barrier_setup}-{flow_ls}.csv"
            results_df.to_csv(out_dir / filename, index=False)
                
        except Exception as e:
            print(f"Solver failed to run for Barrier Setup: {barrier_setup}, Flow: {flow_ls}. Error: {e}")