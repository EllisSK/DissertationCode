# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd
from pandas.core.api import DataFrame as DataFrame

from .basemodel import BaseModel

class AdvancedSluiceModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _equation(self, upstream_depth, gap_size):
        coeff_contraction = np.pi / (np.pi + 2)
        coeff_velocity = 0.99

        coeff_discharge = coeff_contraction * coeff_velocity
        vc_depth = coeff_contraction * gap_size

        head = upstream_depth - vc_depth

        return coeff_discharge * np.sqrt(2 * 9.80665 * head)
    
    def _create_model_dataframe(self, df: DataFrame) -> DataFrame:
        df = super()._create_model_dataframe(df)

        df["Total Head (m)"] = (df["Mean Upstream Depth (mm)"] / 1000) + ((df["Upstream Velocity (m/s)"] ** 2) / (2 * 9.80665))
        df["Sluice Gap (m)"] = (df["Barrier Setup"].str.split("-", n=1).str[0].astype(int) / 1000)
        return df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Modelled Flow (m3/s)"] = self._equation(df["Mean Upstream Depth (mm)"]/1000, df["Sluice Gap (m)"])

        return df
    
class SimpleSluiceModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _equation(self, upstream_depth, gap_size):
        submerged = False
        
        if not submerged:
            coeff_contraction = np.pi / (np.pi + 2)
            coeff_velocity = 0.99

            coeff_discharge = coeff_contraction * coeff_velocity
            vc_depth = coeff_contraction * gap_size

            head = upstream_depth - vc_depth

            return coeff_discharge * np.sqrt(2 * 9.80665 * head)
        else:
            pass
    
    def _create_model_dataframe(self, df: DataFrame) -> DataFrame:
        df = super()._create_model_dataframe(df)

        df["Total Head (m)"] = (df["Mean Upstream Depth (mm)"] / 1000) + ((df["Upstream Velocity (m/s)"] ** 2) / (2 * 9.80665))
        df["Sluice Gap (m)"] = (df["Barrier Setup"].str.split("-", n=1).str[0].astype(int) / 1000)
        return df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Modelled Flow (m3/s)"] = self._equation(df["Mean Upstream Depth (mm)"]/1000, df["Sluice Gap (m)"])

        return df