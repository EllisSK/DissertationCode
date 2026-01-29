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

    def write_report(self, report_path: Path):
        pass

    def create_figure(self):
        pass

    @abstractmethod
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df["Operation Mode"] == "Sluice"]

        df["Set Flow (l/s)"] = pd.to_numeric(df["Set Flow (l/s)"], errors="coerce")
        df["Mean Upstream Depth (mm)"] = pd.to_numeric(df["Mean Upstream Depth (mm)"], errors="coerce")

        df["Flow (m3/s)"] = df["Set Flow (l/s)"] / 1000
        df["Upstream Velocity (m/s)"] = df["Flow (m3/s)"] / (df["Mean Upstream Depth (mm)"] / 1000)

        return df