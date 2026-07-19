# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Sluice-gate discharge models for flow under the bottom plank of the barrier."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.constants import GRAVITY

from .basemodel import BaseModel


def _add_sluice_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Operation Mode"] == "Sluice"]

    df["Sluice Gap (m)"] = (df["Barrier Setup"].str.split("-", n=1).str[0].astype(int) / 1000)
    return df


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
        return _add_sluice_geometry(super()._create_model_dataframe(df))

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

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        if not self.fitted:
            raise Exception("Model hasn't been fit yet!")

        self._write_report_file(
            report_directory,
            "Simple Sluice Model Report",
            [
                f"Optimised Coefficient: {self.popt[0]}",
                f"Optimised Coefficient Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}",
            ],
        )


class AdvancedSluiceModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)

        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X):
        upstream_depth, gap_size = X

        coeff_contraction = np.pi / (np.pi + 2)
        coeff_velocity = 0.98

        coeff_discharge = coeff_contraction * coeff_velocity
        vc_depth = coeff_contraction * gap_size

        head = upstream_depth - vc_depth

        return coeff_discharge * gap_size * np.sqrt(2 * GRAVITY * head)

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_sluice_geometry(super()._create_model_dataframe(df))

    def predict(self, X):
        flow = self._equation(X)
        return flow

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = self.predict((df["Upstream Head (m)"], df["Sluice Gap (m)"]))

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        self._write_report_file(report_directory, "Advanced Sluice Model Report")
