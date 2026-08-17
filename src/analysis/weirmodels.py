# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Sharp-crested weir discharge models for flow over the top of the barrier."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.constants import GRAVITY, PLANK_1_HEIGHT, PLANK_2_HEIGHT, PLANK_3_HEIGHT

from .basemodel import BaseModel


def _add_weir_geometry(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["Operation Mode"] == "Weir")]

    split_data = df["Barrier Setup"].str.split("-", expand=True).astype(int)
    is_gap1_zero = split_data[0] == 0
    is_gap2_zero = split_data[1] == 0
    is_gap3_zero = split_data[2] == 0

    # The crest sits on top of however many planks rest directly on the ones below
    df["Weir Height (m)"] = (
        (is_gap1_zero * PLANK_1_HEIGHT)
        + ((is_gap1_zero & is_gap2_zero) * PLANK_2_HEIGHT)
        + ((is_gap1_zero & is_gap2_zero & is_gap3_zero) * PLANK_3_HEIGHT)
    )
    df["Head on Weir (m)"] = df["Upstream Head (m)"] - df["Weir Height (m)"]

    df = df[(df["Head on Weir (m)"] > 0)]

    return df


class SimpleWeirModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0
        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X, coeff):
        return coeff * np.power(X, 1.5)

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_weir_geometry(super()._create_model_dataframe(df))

    def predict(self, X):
        if self.fitted:
            flow = self._equation(X, self.optimal)
            return flow
        else:
            raise Exception("Model hasn't been fit yet!")

    def fit(self):
        df = self.df

        x_data = df["Head on Weir (m)"]

        y_data = df["Flow (m3/s)"]

        self.popt, self.pcov = curve_fit(self._equation, x_data, y_data)

        self.optimal = self.popt[0]
        self.fitted = True

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = self.predict((df["Head on Weir (m)"]))

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        if not self.fitted:
            raise Exception("Model hasn't been fit yet!")

        self._write_report_file(
            report_directory,
            "Simple Weir Model Report",
            [
                f"Optimised Coefficient: {self.popt[0]}",
                f"Optimised Coefficient Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}",
            ],
        )


class AdvancedWeirModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X):
        h, p = X

        coeff_discharge = (np.pi / (np.pi + 2)) * 0.98

        return (2 / 3) * coeff_discharge * np.sqrt(2 * GRAVITY) * np.power(h, 1.5)

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return _add_weir_geometry(super()._create_model_dataframe(df))

    def predict(self, X):
        flow = self._equation(X)
        return flow

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = self.predict((df["Head on Weir (m)"], df["Weir Height (m)"]))

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        self._write_report_file(report_directory, "Advanced Weir Model Report")
