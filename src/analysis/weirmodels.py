# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
import numpy as np
import pandas as pd

from .basemodel import BaseModel
from scipy.optimize import curve_fit

class SimpleWeirModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0
        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X, coeff):
        return coeff * np.power(X, 1.5)

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df = df[(df["Operation Mode"] == "Weir")]

        split_data = df["Barrier Setup"].str.split("-", expand=True).astype(int)
        is_gap1_zero = split_data[0] == 0
        is_gap2_zero = split_data[1] == 0
        is_gap3_zero = split_data[2] == 0

        df["Weir Height (m)"] = ((is_gap1_zero * 0.2) + ((is_gap1_zero & is_gap2_zero) * 0.1) + ((is_gap1_zero & is_gap2_zero & is_gap3_zero) * 0.1))
        df["Head on Weir (m)"] = df["Upstream Head (m)"] - df["Weir Height (m)"]

        df = df[(df["Head on Weir (m)"] > 0)]

        return df
    
    def predict(self, X):
        if self.fitted:
            flow = self._equation(X, self.optimal)
            return flow
        else:
            raise Exception("Model hasn't been fit yet!")
        
    def fit(self):
        df = self.df 
        
        x_data = (
            df["Head on Weir (m)"]
        )

        y_data = df["Flow (m3/s)"]

        self.popt, self.pcov = curve_fit(self._equation, x_data, y_data)
         
        self.optimal = self.popt[0]
        self.fitted = True

    def write_report(self, report_directory: Path):
        file_path = report_directory / f"{self.name}.txt"
        
        if self.fitted:
            with open(file_path, "w") as f:
                f.write(f"Simple Weir Model Report\n")
                f.write(f"Optimised Coefficient: {self.popt[0]}\n")
                f.write(f"Optimised Coefficient Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}\n")
        else:
            raise Exception("Model hasn't been fit yet!")

class AdvancedWeirModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0
