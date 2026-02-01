# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
import numpy as np
import pandas as pd

from .basemodel import BaseModel
from scipy.optimize import curve_fit

class SimpleWeirModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0

    def _equation(self, head, coeff):
        return coeff * np.power(head, 3/2)

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df = df[df["Operation Mode"] == "Weir"]

        df["Weir Height (m)"] = 0.4
        return df

class AdvancedWeirModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.fitted = False
        self.optimal = 0.0
