# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Validation of the numerical solver against the measured depth profiles."""

import numpy as np
import pandas as pd
from pathlib import Path

from . import objective


def _write_metric_block(f, title: str, observed: pd.Series, predicted: pd.Series):
    rmse, mae, bias, var, corr, kge, r2 = objective.all_metrics(observed, predicted)

    f.write(f"{title}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"Absolute Bias: {bias}\n")
    f.write(f"Variability Ratio: {var}\n")
    f.write(f"Correlation: {corr}\n")
    f.write(f"KGE: {kge}\n")
    f.write(f"R Squared: {r2}\n")


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

    file_path = report_directory / "FrictionValidationReport.txt"

    with open(file_path, "w") as f:
        _write_metric_block(f, "Friction Validation Report", observed, predicted)


def validate_barrier_experiments(df: pd.DataFrame, report_directory: Path):
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

    file_path = report_directory / "BarrierValidationReport.txt"

    with open(file_path, "w") as f:
        _write_metric_block(f, "Barrier Validation Report - All", pd.Series(observed_all), pd.Series(predicted_all))
        f.write("\n")
        _write_metric_block(f, "Barrier Validation Report - Upstream", pd.Series(observed_upstream), pd.Series(predicted_upstream))
        f.write("\n")
        _write_metric_block(f, "Barrier Validation Report - Downstream", pd.Series(observed_downstream), pd.Series(predicted_downstream))
