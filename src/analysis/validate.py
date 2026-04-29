import numpy as np
import pandas as pd
from pathlib import Path
from . import objective

def validate_friction_experiments(report_directory: Path):
    path = Path("data/ManningsNExperiments.csv")
    df = pd.read_csv(path)
    
    observed_list = []
    predicted_list = []
    
    grouped = df.groupby(["Set Flow (l/s)", "Incline (%)"])
    
    for (flow_ls, incline_pct), group in grouped:
        filename = f"{int(incline_pct * 10)}-{int(flow_ls)}.csv"
        sim_path = Path("exports/numerical/friction") / filename
        
        if sim_path.exists():
            sim_df = pd.read_csv(sim_path)
            
            for _, row in group.iterrows():
                obs_x = row["X Position (mm)"] / 1000.0
                obs_depth = row["Depth (mm)"]
                
                pred_depth = np.interp(obs_x, sim_df["X Position (m)"], sim_df["Depth (mm)"])
                
                observed_list.append(obs_depth)
                predicted_list.append(pred_depth)
                
    observed = pd.Series(observed_list)
    predicted = pd.Series(predicted_list)
    
    rmse = objective.rmse(observed, predicted)
    mae = objective.mae(observed, predicted)
    bias = objective.bias(observed, predicted)
    var = objective.variability(observed, predicted)
    corr = objective.correlation(observed, predicted)
    kge = objective._kge(observed, predicted)
    r2 = objective.r2(observed, predicted)
    
    file_path = report_directory / "FrictionValidationReport.txt"
    
    with open(file_path, "w") as f:
        f.write("Friction Validation Report\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"Absolute Bias: {bias}\n")
        f.write(f"Variability Ratio: {var}\n")
        f.write(f"Correlation: {corr}\n")
        f.write(f"KGE: {kge}\n")
        f.write(f"R Squared: {r2}\n")

def validate_barrier_experiments(report_directory: Path):
    path = Path("data/BarrierExperiments.csv")
    df = pd.read_csv(path)
    
    observed_all = []
    predicted_all = []
    
    observed_upstream = []
    predicted_upstream = []
    
    observed_downstream = []
    predicted_downstream = []
    
    grouped = df.groupby(["Barrier Setup", "Set Flow (l/s)"])
    
    for (barrier_setup, flow_ls), group in grouped:
        filename = f"{barrier_setup}-{flow_ls}.csv"
        sim_path = Path("exports/numerical/barriers") / filename
        
        if sim_path.exists():
            sim_df = pd.read_csv(sim_path)
            
            for _, row in group.iterrows():
                obs_x_mm = row["X Position (mm)"]
                obs_x_m = obs_x_mm / 1000.0
                obs_depth = row["Depth (mm)"]
                
                pred_depth = np.interp(obs_x_m, sim_df["X Position (m)"], sim_df["Depth (mm)"])
                
                observed_all.append(obs_depth)
                predicted_all.append(pred_depth)
                
                if obs_x_mm < 5000:
                    observed_upstream.append(obs_depth)
                    predicted_upstream.append(pred_depth)
                else:
                    observed_downstream.append(obs_depth)
                    predicted_downstream.append(pred_depth)
                
    observed_all = pd.Series(observed_all)
    predicted_all = pd.Series(predicted_all)
    
    observed_upstream = pd.Series(observed_upstream)
    predicted_upstream = pd.Series(predicted_upstream)
    
    observed_downstream = pd.Series(observed_downstream)
    predicted_downstream = pd.Series(predicted_downstream)
    
    file_path = report_directory / "BarrierValidationReport.txt"
    
    with open(file_path, "w") as f:
        f.write("Barrier Validation Report - All\n")
        f.write(f"RMSE: {objective.rmse(observed_all, predicted_all)}\n")
        f.write(f"MAE: {objective.mae(observed_all, predicted_all)}\n")
        f.write(f"Absolute Bias: {objective.bias(observed_all, predicted_all)}\n")
        f.write(f"Variability Ratio: {objective.variability(observed_all, predicted_all)}\n")
        f.write(f"Correlation: {objective.correlation(observed_all, predicted_all)}\n")
        f.write(f"KGE: {objective._kge(observed_all, predicted_all)}\n")
        f.write(f"R Squared: {objective.r2(observed_all, predicted_all)}\n")
        f.write("\n")
        
        f.write("Barrier Validation Report - Upstream\n")
        f.write(f"RMSE: {objective.rmse(observed_upstream, predicted_upstream)}\n")
        f.write(f"MAE: {objective.mae(observed_upstream, predicted_upstream)}\n")
        f.write(f"Absolute Bias: {objective.bias(observed_upstream, predicted_upstream)}\n")
        f.write(f"Variability Ratio: {objective.variability(observed_upstream, predicted_upstream)}\n")
        f.write(f"Correlation: {objective.correlation(observed_upstream, predicted_upstream)}\n")
        f.write(f"KGE: {objective._kge(observed_upstream, predicted_upstream)}\n")
        f.write(f"R Squared: {objective.r2(observed_upstream, predicted_upstream)}\n")
        f.write("\n")
        
        f.write("Barrier Validation Report - Downstream\n")
        f.write(f"RMSE: {objective.rmse(observed_downstream, predicted_downstream)}\n")
        f.write(f"MAE: {objective.mae(observed_downstream, predicted_downstream)}\n")
        f.write(f"Absolute Bias: {objective.bias(observed_downstream, predicted_downstream)}\n")
        f.write(f"Variability Ratio: {objective.variability(observed_downstream, predicted_downstream)}\n")
        f.write(f"Correlation: {objective.correlation(observed_downstream, predicted_downstream)}\n")
        f.write(f"KGE: {objective._kge(observed_downstream, predicted_downstream)}\n")
        f.write(f"R Squared: {objective.r2(observed_downstream, predicted_downstream)}\n")