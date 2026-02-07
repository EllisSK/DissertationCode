# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
import numpy as np
import pandas as pd

from .basemodel import BaseModel
from scipy.optimize import curve_fit

class SimpleSluiceModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0
        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X, coeff):
        upstream_depth, gap_size = X
        return coeff * gap_size * np.sqrt(upstream_depth)
    
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df = df[df["Operation Mode"] == "Sluice"]

        df["Sluice Gap (m)"] = (df["Barrier Setup"].str.split("-", n=1).str[0].astype(int) / 1000)
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
            df["Upstream Head (m)"], 
            df["Sluice Gap (m)"]
        )

        y_data = df["Flow (m3/s)"]

        self.popt, self.pcov = curve_fit(self._equation, x_data, y_data)
         
        self.optimal = self.popt[0]
        self.fitted = True

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = self.predict((df["Upstream Head (m)"], df["Sluice Gap (m)"]))
        
        observed = df["Flow (m3/s)"]
        predicted = df["Predicted"]
        
        rmse = self._rmse(observed, predicted)
        mae = self._mae(observed, predicted)
        bias = self._bias(observed, predicted)
        var = self._variability(observed, predicted)
        corr = self._correlation(observed, predicted)
        kge = self._kge(observed, predicted)

        return rmse, mae, bias, var, corr, kge

    def write_report(self, report_directory: Path):
        file_path = report_directory / f"{self.name}.txt"
        
        if self.fitted:
            rmse, mae, bias, var, corr, kge = self._calculate_objective_functions(self.df)
            
            with open(file_path, "w") as f:
                f.write(f"Simple Sluice Model Report\n")
                f.write(f"Optimised Coefficient: {self.popt[0]}\n")
                f.write(f"Optimised Coefficient Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}\n")
                f.write(f"RMSE: {rmse}\n")
                f.write(f"MAE: {mae}\n")
                f.write(f"Absolute Bias: {bias}\n")
                f.write(f"Variability Ratio: {var}\n")
                f.write(f"Correlation: {corr}\n")
                f.write(f"KGE: {kge}\n")
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
    
    def predict(self, df: pd.DataFrame):
        pass