# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd

from openpyxl import load_workbook
from pathlib import Path
from tqdm import tqdm


def read_lab_data() -> pd.DataFrame:
    path = Path("data/LabData.xlsx")

    column_names = [
        "Barrier Setup",
        "Operation Mode",
        "Set Flow (l/s)",
        "Mean Upstream Depth (mm)",
        "Mean Downstream Depth (mm)",
        "Mean Vena Contracta Depth (mm)",
    ]

    book = load_workbook(filename=path, read_only=True, data_only=True)
    config_sheet_names = [s for s in book.sheetnames if s.startswith("Config ")]
    data_list = []

    print("Loading Lab Data. Progress:")
    for sheet_name in tqdm(config_sheet_names):
        sheet = book[sheet_name]

        raw_us_values = [cell.value for row in sheet["D11:D16"] for cell in row]
        us_numbers = [v for v in raw_us_values if isinstance(v, (int, float))]
        if us_numbers:
            avg_us_val = sum(us_numbers) / len(us_numbers)
        else:
            avg_us_val = None

        raw_ds_values = [cell.value for row in sheet["D17:D22"] for cell in row]
        ds_numbers = [v for v in raw_ds_values if isinstance(v, (int, float))]
        if ds_numbers:
            avg_ds_val = sum(ds_numbers) / len(ds_numbers)
        else:
            avg_ds_val = None

        raw_vc_values = [cell.value for row in sheet["D23:D25"] for cell in row]
        vc_numbers = [v for v in raw_vc_values if isinstance(v, (int, float))]
        if vc_numbers:
            avg_vc_val = sum(vc_numbers) / len(vc_numbers)
        else:
            avg_vc_val = None

        data_list.append(
            {
                "Barrier Setup": f"{sheet['F2'].value}-{sheet['F3'].value}-{sheet['F4'].value}",
                "Operation Mode": sheet["F5"].value,
                "Set Flow (l/s)": sheet["B2"].value,
                "Mean Upstream Depth (mm)": avg_us_val,
                "Mean Downstream Depth (mm)": avg_ds_val,
                "Mean Vena Contracta Depth (mm)": avg_vc_val,
            }
        )

    return pd.DataFrame(data_list, columns=column_names)

def read_friction_data() -> pd.DataFrame:
    path = Path("data/ManningsNExperiments.csv")

    df = pd.read_csv(path)

    df = df.groupby(["Set Flow (l/s)", "Incline (%)", "X Position (mm)"])["Depth (mm)"].mean().reset_index()

    return df

def analyse_friction_data(data: pd.DataFrame) -> pd.DataFrame:
    data["AOD (m)"] = ((10000 - data["X Position (mm)"]) / 1000) * (data["Incline (%)"] / 100)
    data["Velocity Head (m)"] = np.power((data["Set Flow (l/s)"] / data["Depth (mm)"]), 2) / (2 * 9.81)
    data["Depth (m)"] = data["Depth (mm)"] / 1000

    data["Total Head (m)"] = data["AOD (m)"] + data["Velocity Head (m)"] + data["Depth (m)"]

    data["X Position (m)"] = data["X Position (mm)"] / 1000

    def get_head_slope(group):
        slope, _ = np.polyfit(group["X Position (m)"], group["Total Head (m)"], 1)
        return slope
    
    slope_data = data.groupby(["Set Flow (l/s)", "Incline (%)"]).apply(get_head_slope).reset_index(name="Free Surface Slope")

    print(slope_data)

    data = data.merge(slope_data, on=["Set Flow (l/s)", "Incline (%)"], how="left")
    data["Sf"] = -data["Free Surface Slope"]
    data["P (m)"] = 2 + (2 * data["Depth (m)"])

    data["Composite n"] = np.sqrt((data["Sf"] * np.power(data["Depth (m)"], 10/3)) / (np.power(data["Set Flow (l/s)"] / 1000, 2) * np.power(data["P (m)"], 4/3)))
    
    print(data)

    x_vals = (2 * data["Depth (m)"]) / 2
    y_vals = (data["P (m)"] * np.power(data["Composite n"], 1.5)) / 2
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    
    print(f"Bed Manning's n: {np.power(intercept, 2/3)}")
    print(f"Wall Manning's n: {np.power(slope, 2/3)}")

    return data

def read_lab_data_for_monte_carlo() -> pd.DataFrame:
    raise NotImplementedError