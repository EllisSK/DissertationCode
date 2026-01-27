# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pathlib import Path
from typing import Callable


def create_model_report(
    model: Callable, data: pd.DataFrame, report_name: str, *col_names: str
):
    report_directory = Path(f"exports/reports")
    report_directory.mkdir(parents=True, exist_ok=True)

    report_path = report_directory / f"{report_name}.txt"

    flow_col = col_names[0]
    other_cols = col_names[1:]

    flow_data = data[flow_col].values
    other_data = [data[col].values for col in other_cols]

    def fit_func(X, param):
        return model(*X, param)

    popt, pcov = curve_fit(fit_func, other_data, flow_data)
    optimised_value = popt[0]

    modelled_flow = fit_func(other_data, optimised_value)
    r2 = r2_score(flow_data, modelled_flow)

    with open(report_path, "w") as f:
        f.write(f"Model Optimisation Report: {report_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Function: {model.__name__}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Optimised Fitted Value: {optimised_value}\n")
        f.write("-" * 40 + "\n")
        f.write(f"R Squared Score: {r2}")


def weir_equation(H, Cd):
    return Cd * (2 * np.sqrt(2 * 9.80665 * (H**3))) / 3


def sluice_equation(H, A, Cd):
    return Cd * A * np.sqrt(2 * 9.80665 * H)


def orifice_equation(H, A, Cd):
    return Cd * A * np.sqrt(2 * 9.80665 * H)
