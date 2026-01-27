# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import pathlib
import pandas as pd
from openpyxl import load_workbook


def read_lab_data(path: pathlib.Path) -> pd.DataFrame:
    column_names = [
        "Barrier Setup",
        "Operation Mode",
        "Set Flow (l/s)",
        "Mean Upstream Depth (mm)",
        "Mean Downstream Depth (mm)",
        "Mean Vena Contracta Depth (mm)"
    ]

    book = load_workbook(filename=path, read_only=True, data_only=True)
    config_sheet_names = [s for s in book.sheetnames if s.startswith("Config ")]
    data_list = []

    for sheet_name in config_sheet_names:
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
        
        data_list.append({
            "Barrier Setup" : f"{sheet["F2"].value}-{sheet["F3"].value}-{sheet["F4"].value}",
            "Operation Mode" : sheet["F5"].value,
            "Set Flow (l/s)" : sheet["B2"].value,
            "Mean Upstream Depth (mm)" : avg_us_val,
            "Mean Downstream Depth (mm)" : avg_ds_val,
            "Mean Vena Contracta Depth (mm)" : avg_vc_val 
        })

    return pd.DataFrame(data_list, columns=column_names)

def create_sluice_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["Operation Mode"] == "Sluice"]
    df["Flow (m3/s)"] = df["Set Flow (l/s)"] / 1000
    df["Head (m)"] = (df["Mean Upstream Depth (mm)"] - df["Barrier Setup"].str.split('-', n=1).str[0].astype(int)) / 1000
    df["Sluice Area (m2)"] = df["Barrier Setup"].str.split('-', n=1).str[0].astype(int) / 1000
    df["Coefficient of Discharge"] = df["Flow (m3/s)"] / (df["Sluice Area (m2)"] * ((2*9.81*df["Head (m)"])**(1/2)))
    df["H/a"] = df["Head (m)"] / df["Sluice Area (m2)"]
    return df

def create_sluice_weir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice-Weir"]
    return df

def create_sluice_orifice_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice-Orifice"]
    return df

def create_sluice_orifice_weir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice-Orifice-Weir"]
    return df

def create_sluice_orifice_orifice_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice-Orifice-Orifice"]
    return df

def create_sluice_orifice_orifice_weir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice-Orifice-Orifice-Weir"]
    return df

def create_orifice_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice-Orifice-Orifice-Weir"]
    return df

def create_orifice_weir_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice-Orifice-Orifice-Weir"]
    return df