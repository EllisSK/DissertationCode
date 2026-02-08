# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path
import numpy as np
import pandas as pd

from .basemodel import BaseModel
from .sluicemodels import SimpleSluiceModel, AdvancedSluiceModel
from .weirmodels import SimpleWeirModel, AdvancedWeirModel
from .orificemodels import SimpleOrificeModel, AdvancedOrificeModel
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
    
    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = df.apply(
            lambda row: self.predict(row["Barrier Setup"], row["Upstream Head (m)"]), 
            axis=1
        )
        
        observed = df["Flow (m3/s)"]
        predicted = df["Predicted"]
        
        rmse = self._rmse(observed, predicted)
        mae = self._mae(observed, predicted)
        bias = self._bias(observed, predicted)
        var = self._variability(observed, predicted)
        corr = self._correlation(observed, predicted)
        kge = self._kge(observed, predicted)

        return rmse, mae, bias, var, corr, kge

    def write_report(self, report_directory: Path):
        file_path = report_directory / f"{self.name}.txt"
        
        rmse, mae, bias, var, corr, kge = self._calculate_objective_functions(self.df)
        
        with open(file_path, "w") as f:
            f.write(f"Simple Combined Model Report\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"Absolute Bias: {bias}\n")
            f.write(f"Variability Ratio: {var}\n")
            f.write(f"Correlation: {corr}\n")
            f.write(f"KGE: {kge}\n")

    
class SimpleIFCombinedModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)

        self.df = self._create_model_dataframe(lab_data)

        self.sluice = SimpleSluiceModel("SimpleCombinedSluice", lab_data)
        self.orifice = SimpleOrificeModel("SimpleCombinedOrifice", lab_data)
        self.weir = SimpleWeirModel("SImpleCombinedWeir", lab_data)

        self.sluice.fit()
        self.weir.fit()
        self.orifice.fit()

        self.fitted = False

    def _equation(self, X, IF_sluice, IF_orifice, IF_weir):
        sluice_params = X[:2]
        orifice_params = X[2:4]
        weir_params = X[4]

        sluice_flow = self.sluice.predict(sluice_params)
        orifice_flow = self.orifice.predict(orifice_params)
        weir_flow = self.weir.predict(weir_params)
        
        return (IF_sluice*sluice_flow) + (IF_orifice*orifice_flow) + (IF_weir*weir_flow)
    
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
            
    def _create_model_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df =  super()._create_model_dataframe(df)

        df["Params"] = df.apply(
            lambda row: self._params_from_setup(
                row["Barrier Setup"], 
                row["Upstream Head (m)"]
            ), 
            axis=1
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

    def plotting_function(self, upstream_depth, barrier_setup):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth), self.if_sluice, self.if_orifice, self.if_weir)
    
    def predict(self, barrier_setup, upstream_depth):
        return self._equation(self._params_from_setup(barrier_setup, upstream_depth), self.if_sluice, self.if_orifice, self.if_weir)
    
    def _calculate_objective_functions(self, df: pd.DataFrame):
        df = df.copy()
        df["Predicted"] = df.apply(
            lambda row: self.predict(row["Barrier Setup"], row["Upstream Head (m)"]), 
            axis=1
        )
        
        observed = df["Flow (m3/s)"]
        predicted = df["Predicted"]
        
        rmse = self._rmse(observed, predicted)
        mae = self._mae(observed, predicted)
        bias = self._bias(observed, predicted)
        var = self._variability(observed, predicted)
        corr = self._correlation(observed, predicted)
        kge = self._kge(observed, predicted)

        return rmse, mae, bias, var, corr, kge

    def write_report(self, report_directory: Path):
        file_path = report_directory / f"{self.name}.txt"
        
        if self.fitted:
            rmse, mae, bias, var, corr, kge = self._calculate_objective_functions(self.df)
            
            with open(file_path, "w") as f:
                f.write(f"Interaction Factor Combined Model Report\n")
                f.write(f"Optimised Sluice Interaction Factor: {self.popt[0]}\n")
                f.write(f"Optimised Sluice Interaction Factor Standard Deviation: {np.sqrt(np.diag(self.pcov))[0]}\n")
                f.write(f"Optimised Orifice Interaction Factor: {self.popt[1]}\n")
                f.write(f"Optimised Orifice Interaction Factor Standard Deviation: {np.sqrt(np.diag(self.pcov))[1]}\n")
                f.write(f"Optimised Weir Interaction Factor: {self.popt[2]}\n")
                f.write(f"Optimised Weir Interaction Factor Standard Deviation: {np.sqrt(np.diag(self.pcov))[2]}\n")
                f.write(f"RMSE: {rmse}\n")
                f.write(f"MAE: {mae}\n")
                f.write(f"Absolute Bias: {bias}\n")
                f.write(f"Variability Ratio: {var}\n")
                f.write(f"Correlation: {corr}\n")
                f.write(f"KGE: {kge}\n")
        else:
            raise Exception("Model hasn't been fit yet!")
        
class AdvancedCombinedModel(BaseModel):
    def __init__(self, name: str, lab_data: pd.DataFrame) -> None:
        super().__init__(name)
        self.sluice = AdvancedSluiceModel("SimpleCombinedSluice", lab_data)
        self.orifice = AdvancedOrificeModel("SimpleCombinedOrifice", lab_data)
        self.weir = AdvancedWeirModel("SImpleCombinedWeir", lab_data)

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
                
                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, h_weir, plank3_top)
            elif upstream_depth > plank3_bottom:
                #Sluice-Orifice-Orifice
                h_sluice = upstream_depth
                sluice_gap = gap1

                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                hb_orifice2 = upstream_depth - plank2_top
                ht_orifice2 = upstream_depth - plank3_bottom

                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, 0, 0)
            elif upstream_depth > plank2_top:
                #Sluice-Orifice-Weir
                h_sluice = upstream_depth
                sluice_gap = gap1

                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                h_weir = upstream_depth - plank2_top

                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, 0, 0, h_weir, plank2_top)
            elif upstream_depth > plank2_bottom:
                #Sluice-Orifice
                h_sluice = upstream_depth
                sluice_gap = gap1

                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom
             
                return (h_sluice, sluice_gap, hb_orifice1, ht_orifice1, 0, 0, 0, 0)
            elif upstream_depth > plank1_top:
                #Sluice-Weir
                h_sluice = upstream_depth
                sluice_gap = gap1

                h_weir = upstream_depth - plank1_top
                
                return (h_sluice, sluice_gap, 0, 0, 0, 0, h_weir, plank1_top)
            else:
                #Sluice
                h_sluice = upstream_depth
                sluice_gap = gap1
                
                return (h_sluice, sluice_gap, 0, 0, 0, 0, 0, 0)
        else:
            if upstream_depth > plank3_top:
                #Orifice-Orifice-Weir
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                hb_orifice2 = upstream_depth - plank2_top
                ht_orifice2 = upstream_depth - plank3_bottom

                h_weir = upstream_depth - plank3_top

                return (0, 0, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, h_weir, plank3_top) 
            elif upstream_depth > plank3_bottom:
                #Orifice-Orifice
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                hb_orifice2 = upstream_depth - plank2_top
                ht_orifice2 = upstream_depth - plank3_bottom

                return (0, 0, hb_orifice1, ht_orifice1, hb_orifice2, ht_orifice2, 0, 0)
            elif upstream_depth > plank2_top:
                #Orifice-Weir
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                h_weir = upstream_depth - plank2_top

                return (0, 0, hb_orifice1, ht_orifice1, 0, 0, h_weir, plank2_top)
            elif upstream_depth > plank2_bottom:
                #Orifice
                hb_orifice1 = upstream_depth - plank1_top
                ht_orifice1 = upstream_depth - plank2_bottom

                return (0, 0, hb_orifice1, ht_orifice1, 0, 0, 0, 0)
            else:
                #Weir
                h_weir = upstream_depth - plank1_top

                return (0, 0, 0, 0, 0, 0, h_weir, plank1_top)

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
            axis=1
        )
        
        observed = df["Flow (m3/s)"]
        predicted = df["Predicted"]
        
        rmse = self._rmse(observed, predicted)
        mae = self._mae(observed, predicted)
        bias = self._bias(observed, predicted)
        var = self._variability(observed, predicted)
        corr = self._correlation(observed, predicted)
        kge = self._kge(observed, predicted)

        return rmse, mae, bias, var, corr, kge

    def write_report(self, report_directory: Path):
        file_path = report_directory / f"{self.name}.txt"
        
        rmse, mae, bias, var, corr, kge = self._calculate_objective_functions(self.df)
        
        with open(file_path, "w") as f:
            f.write(f"Simple Combined Model Report\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"Absolute Bias: {bias}\n")
            f.write(f"Variability Ratio: {var}\n")
            f.write(f"Correlation: {corr}\n")
            f.write(f"KGE: {kge}\n")
