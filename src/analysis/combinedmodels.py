# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

"""Combined discharge models superposing sluice, orifice and weir flow through the barrier."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.constants import PLANK_1_HEIGHT, PLANK_2_HEIGHT, PLANK_3_HEIGHT

from .basemodel import BaseModel
from .orificemodels import AdvancedOrificeModel, SimpleOrificeModel
from .sluicemodels import AdvancedSluiceModel, SimpleSluiceModel
from .weirmodels import AdvancedWeirModel, SimpleWeirModel


def barrier_flow_params(barrier_setup: str, upstream_depth: float) -> tuple:
    """Partition the upstream head across the flow paths through the barrier.

    Depending on how far up the barrier the water sits, flow passes under the
    bottom plank (sluice), through the inter-plank gaps (orifices) and over the
    top of the highest submerged plank (weir).

    Returns an 8-tuple:
    (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, h_weir, weir_crest)
    where inactive flow paths are zeroed and heads are relative to the relevant opening.
    """
    split_data = list(map(int, barrier_setup.split("-")))

    gap1 = split_data[0] / 1000
    gap2 = split_data[1] / 1000
    gap3 = split_data[2] / 1000

    plank1_top = gap1 + PLANK_1_HEIGHT
    plank2_bottom = plank1_top + gap2
    plank2_top = plank2_bottom + PLANK_2_HEIGHT
    plank3_bottom = plank2_top + gap3
    plank3_top = plank3_bottom + PLANK_3_HEIGHT

    orifice1 = (upstream_depth - plank1_top, upstream_depth - plank2_bottom)
    orifice2 = (upstream_depth - plank2_top, upstream_depth - plank3_bottom)

    if gap1 != 0:
        sluice = (upstream_depth, gap1)

        if upstream_depth > plank3_top:
            # Sluice-Orifice-Orifice-Weir
            return (
                *sluice,
                *orifice1,
                *orifice2,
                upstream_depth - plank3_top,
                plank3_top,
            )
        elif upstream_depth > plank3_bottom:
            # Sluice-Orifice-Orifice
            return (*sluice, *orifice1, *orifice2, 0, 0)
        elif upstream_depth > plank2_top:
            # Sluice-Orifice-Weir
            return (*sluice, *orifice1, 0, 0, upstream_depth - plank2_top, plank2_top)
        elif upstream_depth > plank2_bottom:
            # Sluice-Orifice
            return (*sluice, *orifice1, 0, 0, 0, 0)
        elif upstream_depth > plank1_top:
            # Sluice-Weir
            return (*sluice, 0, 0, 0, 0, upstream_depth - plank1_top, plank1_top)
        else:
            # Sluice
            return (*sluice, 0, 0, 0, 0, 0, 0)
    else:
        if upstream_depth > plank3_top:
            # Orifice-Orifice-Weir
            return (0, 0, *orifice1, *orifice2, upstream_depth - plank3_top, plank3_top)
        elif upstream_depth > plank3_bottom:
            # Orifice-Orifice
            return (0, 0, *orifice1, *orifice2, 0, 0)
        elif upstream_depth > plank2_top:
            # Orifice-Weir
            return (0, 0, *orifice1, 0, 0, upstream_depth - plank2_top, plank2_top)
        elif upstream_depth > plank2_bottom:
            # Orifice
            return (0, 0, *orifice1, 0, 0, 0, 0)
        else:
            # Weir
            return (0, 0, 0, 0, 0, 0, upstream_depth - plank1_top, plank1_top)


class SimpleCombinedModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.sluice = SimpleSluiceModel("SimpleCombinedSluice", lab_data)
        self.orifice = SimpleOrificeModel("SimpleCombinedOrifice", lab_data)
        self.weir = SimpleWeirModel("SimpleCombinedWeir", lab_data)

        self.sluice.fit()
        self.weir.fit()
        self.orifice.fit()

        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X):
        sluice_params = X[:2]
        orifice1_params = X[2:4]
        orifice2_params = X[4:6]
        weir_params = X[6]

        sluice_flow = self.sluice.predict(sluice_params)
        orifice1_flow = self.orifice.predict(orifice1_params)
        orifice2_flow = self.orifice.predict(orifice2_params)
        weir_flow = self.weir.predict(weir_params)

        return sluice_flow + orifice1_flow + orifice2_flow + weir_flow

    def _params_from_setup(self, barrier_setup, upstream_depth):
        return barrier_flow_params(barrier_setup, upstream_depth)[:7]

    def plotting_function(self, upstream_depth, barrier_setup):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth))

    def predict(self, barrier_setup, upstream_depth):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth))

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return super()._create_model_dataframe(df)

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = df.apply(
            lambda row: self.predict(row["Barrier Setup"], row["Upstream Head (m)"]),
            axis=1,
        )

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        self._write_report_file(report_directory, "Simple Combined Model Report")


class SimpleIFCombinedModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)

        self.df = self._create_model_dataframe(lab_data)

        self.sluice = SimpleSluiceModel("SimpleCombinedSluice", lab_data)
        self.orifice = SimpleOrificeModel("SimpleCombinedOrifice", lab_data)
        self.weir = SimpleWeirModel("SimpleCombinedWeir", lab_data)

        self.sluice.fit()
        self.weir.fit()
        self.orifice.fit()

        self.fitted = False
        self.optimal = np.zeros(3)

    def _equation(self, X, IF_sluice, IF_orifice, IF_weir):
        sluice_params = X[:2]
        orifice_params = X[2:4]
        weir_params = X[4]

        sluice_flow = self.sluice.predict(sluice_params)
        orifice_flow = self.orifice.predict(orifice_params)
        weir_flow = self.weir.predict(weir_params)

        return (
            (IF_sluice * sluice_flow)
            + (IF_orifice * orifice_flow)
            + (IF_weir * weir_flow)
        )

    def _params_from_setup(self, barrier_setup, upstream_depth):
        return barrier_flow_params(barrier_setup, upstream_depth)[:7]

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._create_model_dataframe(df)

        df["Params"] = df.apply(
            lambda row: self._params_from_setup(
                row["Barrier Setup"], row["Upstream Head (m)"]
            ),
            axis=1,
        )

        return df

    def fit(self):
        df = self.df

        x_data = np.array(df["Params"].tolist()).T

        y_data = df["Flow (m3/s)"]

        self.popt, self.pcov = curve_fit(self._equation, x_data, y_data)

        self.fitted = True

        self.if_sluice = self.popt[0]
        self.if_orifice = self.popt[1]
        self.if_weir = self.popt[2]
        self.optimal = self.popt

    def plotting_function(self, upstream_depth, barrier_setup):
        return self._equation(
            self._params_from_setup(barrier_setup, upstream_depth),
            self.if_sluice,
            self.if_orifice,
            self.if_weir,
        )

    def predict(self, barrier_setup, upstream_depth):
        return self._equation(
            self._params_from_setup(barrier_setup, upstream_depth),
            self.if_sluice,
            self.if_orifice,
            self.if_weir,
        )

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = df.apply(
            lambda row: self.predict(row["Barrier Setup"], row["Upstream Head (m)"]),
            axis=1,
        )

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        if not self.fitted:
            raise Exception("Model hasn't been fit yet!")

        coeff_sd = np.sqrt(np.diag(self.pcov))
        self._write_report_file(
            report_directory,
            "Interaction Factor Combined Model Report",
            [
                f"Optimised Sluice Interaction Factor: {self.popt[0]}",
                f"Optimised Sluice Interaction Factor Standard Deviation: {coeff_sd[0]}",
                f"Optimised Orifice Interaction Factor: {self.popt[1]}",
                f"Optimised Orifice Interaction Factor Standard Deviation: {coeff_sd[1]}",
                f"Optimised Weir Interaction Factor: {self.popt[2]}",
                f"Optimised Weir Interaction Factor Standard Deviation: {coeff_sd[2]}",
            ],
        )


class AdvancedCombinedModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.sluice = AdvancedSluiceModel("AdvancedCombinedSluice", lab_data)
        self.orifice = AdvancedOrificeModel("AdvancedCombinedOrifice", lab_data)
        self.weir = AdvancedWeirModel("AdvancedCombinedWeir", lab_data)

        self.df = self._create_model_dataframe(lab_data)

    def _equation(self, X):
        sluice_params = X[:2]
        orifice1_params = X[2:4]
        orifice2_params = X[4:6]
        weir_params = X[6:8]

        sluice_flow = self.sluice.predict(sluice_params)
        orifice1_flow = self.orifice.predict(orifice1_params)
        orifice2_flow = self.orifice.predict(orifice2_params)
        if weir_params[1] > 0:
            weir_flow = self.weir.predict(weir_params)
        else:
            weir_flow = 0

        return sluice_flow + orifice1_flow + orifice2_flow + weir_flow

    def _params_from_setup(self, barrier_setup, upstream_depth):
        return barrier_flow_params(barrier_setup, upstream_depth)

    def plotting_function(self, upstream_depth, barrier_setup):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth))

    def predict(self, barrier_setup, upstream_depth):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth))

    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return super()._create_model_dataframe(df)

    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = df.apply(
            lambda row: self.predict(row["Barrier Setup"], row["Upstream Head (m)"]),
            axis=1,
        )

        return self._metrics(df["Flow (m3/s)"], df["Predicted"])

    def write_report(self, report_directory: Path):
        self._write_report_file(report_directory, "Advanced Combined Model Report")
