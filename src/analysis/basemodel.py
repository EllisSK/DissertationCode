# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import pandas as pd

from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

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

    def create_figure(self):
        pass

    @abstractmethod
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["Set Flow (l/s)"] = pd.to_numeric(df["Set Flow (l/s)"], errors="coerce")
        df["Mean Upstream Depth (mm)"] = pd.to_numeric(df["Mean Upstream Depth (mm)"], errors="coerce")

        df["Flow (m3/s)"] = df["Set Flow (l/s)"] / 1000
        df["Upstream Velocity (m/s)"] = df["Flow (m3/s)"] / (df["Mean Upstream Depth (mm)"] / 1000)
        df["Upstream Head (m)"] = (df["Mean Upstream Depth (mm)"] / 1000)

        return df
    
    @abstractmethod
    def _calculate_objective_functions(self, df: pd.DataFrame) -> tuple:
        pass

    def _rmse(self, observed, predicted):
        return ((predicted - observed) ** 2).mean() ** 0.5

    def _mae(self, observed, predicted):
        return (predicted - observed).abs().mean()

    def _bias(self, observed, predicted):
        return (predicted - observed).mean()

    def _variability(self, observed, predicted):
        return predicted.std() / observed.std()

    def _correlation(self, observed, predicted):
        return observed.corr(predicted)

    def _kge(self, observed, predicted):
        corr = self._correlation(observed, predicted)
        var = self._variability(observed, predicted)
        bias = predicted.mean() / observed.mean()
        return 1 - ((corr - 1) ** 2 + (var - 1) ** 2 + (bias - 1) ** 2) ** 0.5