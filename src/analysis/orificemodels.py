# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Orifice discharge models for flow through a gap between two planks of the barrier."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.constants import GRAVITY

from .basemodel import BaseModel


def _orifice_geometry(row) -> pd.Series:
    barrier_setup = row["Barrier Setup"]
    split_data = list(map(int, barrier_setup.split("-")))

    gap2 = split_data[1] / 1000
    gap3 = split_data[2] / 1000

    # Orifice experiments open exactly one of the two upper gaps; the orifice
    # sits on top of the bottom plank (0.2 m) or the closed two-plank stack (0.3 m)
    if gap2 > 0:
        orifice_size = gap2
        orifice_bottom = 0.2
    else:
        orifice_size = gap3
        orifice_bottom = 0.3
    orifice_top = orifice_bottom + orifice_size

    return pd.Series([orifice_size, orifice_bottom, orifice_top])


def _add_orifice_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["Operation Mode"] == "Orifice")]

    df[["Orifice Size (m)", "Orifice Bottom Height (m)", "Orifice Top Height (m)"]] = df.apply(_orifice_geometry, axis=1)

    df["Depth at Bottom (m)"] = df["Upstream Head (m)"] - df["Orifice Bottom Height (m)"]
    df["Depth at Top (m)"] = df["Upstream Head (m)"] - df["Orifice Top Height (m)"]

    df = df[df["Depth at Top (m)"] > 0]

    return df


class SimpleOrificeModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0
        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X, coeff):
        bottom, top = X
        return coeff * (np.power(bottom, 1.5) - np.power(top, 1.5))

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_orifice_geometry(super()._create_model_dataframe(df))

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

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        if not self.fitted:
            raise Exception("Model hasn't been fit yet!")

        self._write_report_file(
            report_directory,
            "Simple Orifice Model Report",
            [
                f"Optimised Coefficient: {self.popt[0]}",
                f"Optimised Coefficient Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}",
            ],
        )


class AdvancedOrificeModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.df = self._create_model_dataframe(lab_data)

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_orifice_geometry(super()._create_model_dataframe(df))

    def _equation(self, X):
        bottom, top = X

        coeff_contraction = np.pi / (np.pi + 2)
        coeff_velocity = 0.98

        coeff_discharge = coeff_contraction * coeff_velocity

        return (2/3) * coeff_discharge * np.sqrt(2 * GRAVITY) * (np.power(bottom, 1.5) - np.power(top, 1.5))

    def predict(self, X):
        flow = self._equation(X)
        return flow

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = self.predict((df["Depth at Bottom (m)"], df["Depth at Top (m)"]))

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        self._write_report_file(report_directory, "Advanced Orifice Model Report")
