# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
import numpy as np
import pandas as pd

from .basemodel import BaseModel
from scipy.optimize import curve_fit

class SimpleOrificeModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0
        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X, coeff):
        bottom, top = X
        return coeff * (np.power(bottom, 1.5) - np.power(top, 1.5))
    
    def _calculate_orifice_geometry(self, row):
        barrier_setup = row["Barrier Setup"]
        split_data = list(map(int, barrier_setup.split("-")))

        gap2 = split_data[1]
        gap3 = split_data[2]

        if gap2 > 0:
            orifice_size = gap2 / 1000
            orifice_bottom = 0.2
            orifice_top = orifice_bottom + orifice_size
        else:
            orifice_size = gap3 / 1000
            orifice_bottom = 0.3
            orifice_top = orifice_bottom + orifice_size

        return pd.Series([orifice_size, orifice_bottom, orifice_top])

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df = df[(df["Operation Mode"] == "Orifice")]

        df[["Orifice Size (m)", "Orifice Bottom Height (m)", "Orifice Top Height (m)"]] = df.apply(self._calculate_orifice_geometry, axis=1)

        df["Depth at Bottom (m)"] = df["Upstream Head (m)"] - df["Orifice Bottom Height (m)"]
        df["Depth at Top (m)"] = df["Upstream Head (m)"] - df["Orifice Top Height (m)"]

        df = df[df["Depth at Top (m)"] > 0]
        
        return df
    
    def predict(self) -> pd.DataFrame:
        df = self.df

        if self.fitted:
            df["Modelled Flow (m3/s)"] = self._equation((df["Depth at Bottom (m)"], df["Depth at Top (m)"]), self.optimal)
            return df
        else:
            raise Exception("Model hasn't been fit yet!")
        
    def fit(self):
        df = self.df

        x_data = (
            df["Depth at Bottom (m)"], 
            df["Depth at Top (m)"]
        )

        y_data = df["Flow (m3/s)"]

        self.popt, self.pcov = curve_fit(self._equation, x_data, y_data)
         
        self.optimal = self.popt[0]
        self.fitted = True

    def write_report(self, report_directory: Path):
        file_path = report_directory / f"{self.name}.txt"
        
        if self.fitted:
            with open(file_path, "w") as f:
                f.write(f"Simple Orifice Model Report\n")
                f.write(f"Optimised Coefficient: {self.popt[0]}\n")
                f.write(f"Optimised Coefficient Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}\n")
        else:
            raise Exception("Model hasn't been fit yet!")

class AdvancedOrificeModel(BaseModel):
    pass