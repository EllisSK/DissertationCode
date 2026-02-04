# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
import numpy as np
import pandas as pd

from .basemodel import BaseModel
from .sluicemodels import SimpleSluiceModel
from .weirmodels import SimpleWeirModel
from .orificemodels import SimpleOrificeModel
from scipy.optimize import curve_fit

class SimpleCombinedModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.sluice = SimpleSluiceModel("SimpleCombinedSluice", lab_data)
        self.orifice = SimpleOrificeModel("SimpleCombinedOrifice", lab_data)
        self.weir = SimpleWeirModel("SImpleCombinedWeir", lab_data)

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
        
        return sluice_flow + orifice1_flow +  orifice2_flow + weir_flow
    
    def _params_from_setup(self, barrier_setup, upstream_depth):
        split_data = list(map(int, barrier_setup.split("-")))

        gap1 = split_data[0]/1000
        gap2 = split_data[1]/1000
        gap3 = split_data[2]/1000

        plank1 = 0.2
        plank2 = 0.1
        plank3 = 0.1

        plank3_top = gap1 + plank1 + gap2 + plank2 + gap3 + plank3
        plank3_bottom = gap1 + plank1 + gap2 + plank2 + gap3
        plank2_top = gap1 + plank1 + gap2 + plank2
        plank2_bottom = gap1 + plank1 + gap2
        plank1_top = gap1 + plank1
        plank1_bottom = gap1

        if gap1 != 0:
            if upstream_depth > plank3_top:
                #Sluice-Orifice-Orifice-Weir
                h_sluice = upstream_depth
                sluice_gap = gap1

                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                hb_orifice2 = upstream_depth - plank2_top
                ht_orifice2 = upstream_depth - plank3_bottom

                h_weir = upstream_depth - plank3_top
                
                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, h_weir)
            elif upstream_depth > plank3_bottom:
                #Sluice-Orifice-Orifice
                h_sluice = upstream_depth
                sluice_gap = gap1

                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                hb_orifice2 = upstream_depth - plank2_top
                ht_orifice2 = upstream_depth - plank3_bottom

                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, 0)
            elif upstream_depth > plank2_top:
                #Sluice-Orifice-Weir
                h_sluice = upstream_depth
                sluice_gap = gap1

                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                h_weir = upstream_depth - plank2_top

                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, 0, 0, h_weir)
            elif upstream_depth > plank2_bottom:
                #Sluice-Orifice
                h_sluice = upstream_depth
                sluice_gap = gap1

                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom
             
                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, 0, 0, 0)
            elif upstream_depth > plank1_top:
                #Sluice-Weir
                h_sluice = upstream_depth
                sluice_gap = gap1

                h_weir = upstream_depth - plank1_top
                
                return (h_sluice, sluice_gap, 0, 0, 0, 0, h_weir)
            else:
                #Sluice
                h_sluice = upstream_depth
                sluice_gap = gap1
                
                return (h_sluice, sluice_gap, 0, 0, 0, 0, 0)
        else:
            if upstream_depth > plank3_top:
                #Orifice-Orifice-Weir
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                hb_orifice2 = upstream_depth - plank2_top
                ht_orifice2 = upstream_depth - plank3_bottom

                h_weir = upstream_depth - plank3_top

                return (0, 0, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, h_weir) 
            elif upstream_depth > plank3_bottom:
                #Orifice-Orifice
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                hb_orifice2 = upstream_depth - plank2_top
                ht_orifice2 = upstream_depth - plank3_bottom

                return (0, 0, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, 0)
            elif upstream_depth > plank2_top:
                #Orifice-Weir
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                h_weir = upstream_depth - plank2_top

                return (0, 0, hb_orifice1, ht_orifice1, 0, 0, h_weir)
            elif upstream_depth > plank2_bottom:
                #Orifice
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                return (0, 0, hb_orifice1, ht_orifice1, 0, 0, 0)
            else:
                #Weir
                h_weir = upstream_depth - plank1_top

                return (0, 0, 0, 0, 0, 0, h_weir)

    def plotting_function(self, upstream_depth, barrier_setup):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth))
    
    def predict(self, barrier_setup, upstream_depth):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth))
    
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return super()._create_model_dataframe(df)
    
class IFCombinedModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.sluice = SimpleSluiceModel("SimpleCombinedSluice", lab_data)
        self.orifice = SimpleOrificeModel("SimpleCombinedOrifice", lab_data)
        self.weir = SimpleWeirModel("SImpleCombinedWeir", lab_data)

        self.sluice.fit()
        self.weir.fit()
        self.orifice.fit()

    def _equation(self, X, IF_sluice, IF_orifice, IF_weir):
        sluice_params = X[:2]
        orifice_params = X[2:4]
        weir_params = X[4]

        sluice_flow = self.sluice.predict(sluice_params)
        orifice_flow = self.orifice.predict(orifice_params)
        weir_flow = self.weir.predict(weir_params)
        
        return (IF_sluice*sluice_flow) + (IF_orifice*orifice_flow) + (IF_weir*weir_flow)