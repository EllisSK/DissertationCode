# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd

from openpyxl import load_workbook
from pathlib import Path
from tqdm import tqdm

def scan_data_directory():
    data_dir = Path("data/") 

    contents = list(data_dir.iterdir())

    if not (data_dir / "BarrierExperiments.csv").exists():
        if (data_dir / "LabData.xlsx").exists():
            construct_lab_data_csv()
        else:
            # Download from archive
            pass
            
    if not (data_dir / "ManningsNExperiments.csv").exists():
        # Download from archive
        pass

def construct_lab_data_csv():
    path = Path("data/BarrierExperiments.csv")
    
    data = read_lab_data()

    data.to_csv(path, index=False)

def parse_float(val):
    if val is None or str(val).strip().upper() in ("N/A", "NA", ""):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def read_lab_data() -> pd.DataFrame:
    path = Path("data/LabData.xlsx")

    column_names = [
        "Barrier Setup",
        "Operation Mode",
        "Set Flow (l/s)",
        "X Position (mm)",
        "Y Position (mm)",
        "Depth (mm)",
    ]

    book = load_workbook(filename=path, read_only=True, data_only=True)
    config_sheet_names = [s for s in book.sheetnames if s.startswith("Config ")]
    data_list = []

    print("Loading Lab Data. Progress:")
    for sheet_name in tqdm(config_sheet_names):
        sheet = book[sheet_name]

        barrier_setup = "{}-{}-{}".format(sheet["F2"].value, sheet["F3"].value, sheet["F4"].value)
        operation_mode = sheet["F5"].value
        set_flow = parse_float(sheet["B2"].value)

        for row in sheet["B11:D25"]:
            x_pos = parse_float(row[0].value)
            y_pos = parse_float(row[1].value)
            depth = parse_float(row[2].value)

            if x_pos is None or y_pos is None or depth is None:
                continue

            x_pos += 5000

            data_list.append(
                {
                    "Barrier Setup": barrier_setup,
                    "Operation Mode": operation_mode,
                    "Set Flow (l/s)": set_flow,
                    "X Position (mm)": x_pos,
                    "Y Position (mm)": y_pos,
                    "Depth (mm)": depth,
                }
            )

    return pd.DataFrame(data_list, columns=column_names)

def read_barrier_data() -> pd.DataFrame:
    path = Path("data/BarrierExperiments.csv")
    
    df = pd.read_csv(path)

    df = df[df["X Position (mm)"] < 5000]
    
    df_mean = df.groupby(
        ["Barrier Setup", "Operation Mode", "Set Flow (l/s)"], 
        as_index=False
    ).agg(
        **{"Mean Upstream Depth (mm)": ("Depth (mm)", "mean")}
    )
    
    return df_mean

def read_friction_data() -> pd.DataFrame:
    path = Path("data/ManningsNExperiments.csv")

    df = pd.read_csv(path)

    df = df.groupby(["Set Flow (l/s)", "Incline (%)", "X Position (mm)"])["Depth (mm)"].mean().reset_index()

    return df

def analyse_friction_data(data: pd.DataFrame):
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

    data["Composite n"] = np.sqrt((data["Sf"] * np.power(data["Depth (m)"], 10/3)) / (np.power(data["Set Flow (l/s)"] / 1000, 2) * np.power(data["P (m)"], 4/3)))

    x_vals = (2 * data["Depth (m)"]) / 2
    y_vals = (data["P (m)"] * np.power(data["Composite n"], 1.5)) / 2
    slope, intercept = np.polyfit(x_vals, y_vals, 1)

    bed_n = np.power(intercept, 2/3)
    wall_n = np.power(slope, 2/3)

    return bed_n, wall_n

import pandas as pd
from pathlib import Path

def write_friction_report(name: str, path: Path):
    fric = read_friction_data()
    
    data = analyse_friction_data(fric)

    df = pd.DataFrame([data], columns=["Bed", "Wall"])

    df.to_csv(path / name, index=False)

def read_lab_data_for_monte_carlo() -> pd.DataFrame:
    raise NotImplementedError