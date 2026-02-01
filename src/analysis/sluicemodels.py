# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
import numpy as np
import pandas as pd

from .basemodel import BaseModel
from scipy.optimize import curve_fit

class SimpleSluiceModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0

    def _equation(self, X, coeff):
        upstream_depth, gap_size = X
        return coeff * gap_size * np.sqrt(upstream_depth)
    
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df = df[df["Operation Mode"] == "Sluice"]

        df["Total Head (m)"] = (df["Mean Upstream Depth (mm)"] / 1000) + ((df["Upstream Velocity (m/s)"] ** 2) / (2 * 9.80665))
        df["Sluice Gap (m)"] = (df["Barrier Setup"].str.split("-", n=1).str[0].astype(int) / 1000)
        return df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fitted:
            df["Modelled Flow (m3/s)"] = self._equation(df["Mean Upstream Depth (mm)"]/1000, df["Sluice Gap (m)"], self.optimal)
            return df
        else:
            raise Exception("Model hasn't been fit yet!")
        
    def fit(self, df: pd.DataFrame):
        x_data = (
            df["Mean Upstream Depth (mm)"]/1000, 
            df["Sluice Gap (m)"]
        )

        y_data = df["Flow (m3/s)"]

        self.popt, self.pcov = curve_fit(self._equation, x_data, y_data)
         
        self.optimal = self.popt[0]
        self.fitted = True

    def write_report(self, report_directory: Path):
        file_path = report_directory / f"{self.name}.txt"
        
        if self.fitted:
            with open(file_path, "w") as f:
                f.write(f"Simple Sluice Model Report\n")
                f.write(f"Optimised Coefficient: {self.popt[0]}\n")
                f.write(f"Optimised Coefficient Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}\n")
        else:
            raise Exception("Model hasn't been fit yet!")
    
class AdvancedSluiceModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _equation(self, upstream_depth, gap_size):
        submerged = False
        
        if not submerged:
            coeff_contraction = np.pi / (np.pi + 2)
            coeff_velocity = 0.98

            coeff_discharge = coeff_contraction * coeff_velocity
            vc_depth = coeff_contraction * gap_size

            head = upstream_depth - vc_depth

            return coeff_discharge * np.sqrt(2 * 9.80665 * head)
        else:
            pass
    
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df = df[df["Operation Mode"] == "Sluice"]

        df["Sluice Gap (m)"] = (df["Barrier Setup"].str.split("-", n=1).str[0].astype(int) / 1000)
        return df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Modelled Flow (m3/s)"] = self._equation(df["Mean Upstream Depth (mm)"]/1000, df["Sluice Gap (m)"])

        return df