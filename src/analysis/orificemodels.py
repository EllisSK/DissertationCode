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

        gap2 = split_data[1] / 1000
        gap3 = split_data[2] / 1000

        if gap2 > 0:
            orifice_size = gap2
            orifice_bottom = 0.2
            orifice_top = orifice_bottom + orifice_size
        else:
            orifice_size = gap3
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
    
    def predict(self, X):
        if self.fitted:
            flow = self._equation(X, self.optimal)
            return flow
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

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = self.predict((df["Depth at Bottom (m)"], df["Depth at Top (m)"]))
        
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
                f.write(f"Simple Orifice Model Report\n")
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

class AdvancedOrificeModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.df = self._create_model_dataframe(lab_data)

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df = df[(df["Operation Mode"] == "Orifice")]

        df[["Orifice Size (m)", "Orifice Bottom Height (m)", "Orifice Top Height (m)"]] = df.apply(self._calculate_orifice_geometry, axis=1)

        df["Depth at Bottom (m)"] = df["Upstream Head (m)"] - df["Orifice Bottom Height (m)"]
        df["Depth at Top (m)"] = df["Upstream Head (m)"] - df["Orifice Top Height (m)"]

        df = df[df["Depth at Top (m)"] > 0]
        
        return df
    
    def _calculate_orifice_geometry(self, row):
        barrier_setup = row["Barrier Setup"]
        split_data = list(map(int, barrier_setup.split("-")))

        gap2 = split_data[1] / 1000
        gap3 = split_data[2] / 1000

        if gap2 > 0:
            orifice_size = gap2
            orifice_bottom = 0.2
            orifice_top = orifice_bottom + orifice_size
        else:
            orifice_size = gap3
            orifice_bottom = 0.3
            orifice_top = orifice_bottom + orifice_size

        return pd.Series([orifice_size, orifice_bottom, orifice_top])
    
    def _equation(self, X):
        bottom, top = X

        coeff_contraction = np.pi / (np.pi + 2)
        coeff_velocity = 0.98

        coeff_discharge = coeff_contraction * coeff_velocity

        return (2/3) * coeff_discharge * np.sqrt(2 * 9.80665) * (np.power(bottom, 1.5) - np.power(top, 1.5))
    
    def predict(self, X):
        flow = self._equation(X)
        return flow

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = self.predict((df["Depth at Bottom (m)"], df["Depth at Top (m)"]))
        
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
        

        rmse, mae, bias, var, corr, kge = self._calculate_objective_functions(self.df)
        
        with open(file_path, "w") as f:
            f.write(f"Advanced Orifice Model Report\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"Absolute Bias: {bias}\n")
            f.write(f"Variability Ratio: {var}\n")
            f.write(f"Correlation: {corr}\n")
            f.write(f"KGE: {kge}\n")