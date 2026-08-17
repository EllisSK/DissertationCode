# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Common behaviour for the barrier discharge models.

Each model receives the mean upstream depth data, derives its geometry from the
"gap1-gap2-gap3" barrier setup string and predicts the flow through the barrier.
"Simple" models fit an empirical coefficient with least squares; "advanced"
models use analytically derived discharge coefficients and need no fitting.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from . import objective


class BaseModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def _equation(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def write_report(self, report_directory: Path):
        pass

    @abstractmethod
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["Set Flow (l/s)"] = pd.to_numeric(df["Set Flow (l/s)"], errors="coerce")
        df["Mean Upstream Depth (mm)"] = pd.to_numeric(
            df["Mean Upstream Depth (mm)"], errors="coerce"
        )

        df["Flow (m3/s)"] = df["Set Flow (l/s)"] / 1000
        df["Upstream Velocity (m/s)"] = df["Flow (m3/s)"] / (
            df["Mean Upstream Depth (mm)"] / 1000
        )
        df["Upstream Head (m)"] = df["Mean Upstream Depth (mm)"] / 1000

        return df

    @abstractmethod
    def _calculate_objective_functions(self, df: pd.DataFrame) -> tuple:
        pass

    def _metrics(self, observed, predicted) -> tuple:
        return objective.all_metrics(observed, predicted)

    def _write_report_file(
        self, report_directory: Path, title: str, header_lines: list[str] | None = None
    ):
        rmse, mae, bias, var, corr, kge, r2 = self._calculate_objective_functions(
            self.df
        )

        file_path = report_directory / f"{self.name}.txt"

        with open(file_path, "w") as f:
            f.write(f"{title}\n")
            for line in header_lines or []:
                f.write(f"{line}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"Absolute Bias: {bias}\n")
            f.write(f"Variability Ratio: {var}\n")
            f.write(f"Correlation: {corr}\n")
            f.write(f"KGE: {kge}\n")
            f.write(f"R Squared: {r2}\n")
