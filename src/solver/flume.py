# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd
import plotly.express as px

from pathlib import Path
from .solver import simulate, simulate_barrier
from typing import Callable

class Flume:
    def __init__(self, barrier_setup: str | None, set_flow: float, incline: float):
        self.barrier = barrier_setup
        self.flow = set_flow
        self.incline = incline

    def _get_mannings_fn(self) -> Callable:
        path = Path("exports/reports/frictionValues.csv")
        df = pd.read_csv(path)
        
        n_bed = df["Bed"].iloc[0]
        n_wall = df["Wall"].iloc[0]

        def fn(h):
            p_bed = 1.0
            p_wall = 2.0 * h
            p_total = p_bed + p_wall

            n_composite = ((p_bed * (n_bed ** 1.5)) + (p_wall * (n_wall ** 1.5))) / p_total
            
            return n_composite ** (2 / 3)

        return fn

    def _get_barrier_fn(self, barrier_setup: str) -> Callable:
        split_data = list(map(int, barrier_setup.split("-")))
        gap1 = split_data[0] / 1000
        gap2 = split_data[1] / 1000
        gap3 = split_data[2] / 1000

        plank1 = 0.2
        plank2 = 0.1
        plank3 = 0.1

        plank3_top = gap1 + plank1 + gap2 + plank2 + gap3 + plank3
        plank3_bottom = gap1 + plank1 + gap2 + plank2 + gap3
        plank2_top = gap1 + plank1 + gap2 + plank2
        plank2_bottom = gap1 + plank1 + gap2
        plank1_top = gap1 + plank1
        
        coeff_contraction = np.pi / (np.pi + 2)
        coeff_velocity = 0.98
        coeff_discharge = coeff_contraction * coeff_velocity
        g = 9.80665

        def fn(h, ds):
            h_arr = np.atleast_1d(h)
            
            c3t = h_arr > plank3_top
            c3b = h_arr > plank3_bottom
            c2t = h_arr > plank2_top
            c2b = h_arr > plank2_bottom
            c1t = h_arr > plank1_top
            
            if gap1 != 0:
                h_sluice = h_arr
                sluice_gap = np.full_like(h_arr, gap1)
                
                hb_orifice1 = np.where(c2b, h_arr - plank1_top, 0.0)
                ht_orifice1 = np.where(c2b, h_arr - plank2_bottom, 0.0)
                
                hb_orifice2 = np.where(c3b, h_arr - plank2_top, 0.0)
                ht_orifice2 = np.where(c3b, h_arr - plank3_bottom, 0.0)
                
                h_weir = np.select(
                    [c3t, c3b, c2t, c2b, c1t],
                    [h_arr - plank3_top, 0.0, h_arr - plank2_top, 0.0, h_arr - plank1_top],
                    default=0.0
                )
            else:
                h_sluice = np.zeros_like(h_arr)
                sluice_gap = np.zeros_like(h_arr)
                
                hb_orifice1 = np.where(c2b, h_arr - plank1_top, 0.0)
                ht_orifice1 = np.where(c2b, h_arr - plank2_bottom, 0.0)
                
                hb_orifice2 = np.where(c3b, h_arr - plank2_top, 0.0)
                ht_orifice2 = np.where(c3b, h_arr - plank3_bottom, 0.0)
                
                h_weir = np.select(
                    [c3t, c3b, c2t, c2b],
                    [h_arr - plank3_top, 0.0, h_arr - plank2_top, 0.0],
                    default=h_arr - plank1_top
                )

            q_sluice = np.zeros_like(h_arr)
            if gap1 != 0:
                vc_depth = coeff_contraction * sluice_gap
                head_sluice = np.maximum(h_sluice - vc_depth, 0.0)
                q_sluice = coeff_discharge * sluice_gap * np.sqrt(2 * g * head_sluice)
                
            q_orifice1 = (2/3) * coeff_discharge * np.sqrt(2 * g) * (
                np.power(np.maximum(hb_orifice1, 0.0), 1.5) - 
                np.power(np.maximum(ht_orifice1, 0.0), 1.5)
            )
            
            q_orifice2 = (2/3) * coeff_discharge * np.sqrt(2 * g) * (
                np.power(np.maximum(hb_orifice2, 0.0), 1.5) - 
                np.power(np.maximum(ht_orifice2, 0.0), 1.5)
            )
            
            q_weir = (2/3) * coeff_discharge * np.sqrt(2 * g) * np.power(np.maximum(h_weir, 0.0), 1.5)
            
            M_total = np.zeros_like(h_arr)

            # Sluice: jet thickness = Cc * gap, V = q/(Cc*gap)
            if gap1 > 0:
                h_c_sluice = coeff_contraction * sluice_gap
                M_total += np.divide(
                    q_sluice * q_sluice, h_c_sluice,
                    out=np.zeros_like(q_sluice),
                    where=h_c_sluice > 1e-9,
                )

            # Orifice 1 (gap2 between plank1_top and plank2_bottom)
            gap_o1 = plank2_bottom - plank1_top
            if gap_o1 > 0:
                h_c_o1 = coeff_contraction * gap_o1
                M_total += q_orifice1 * q_orifice1 / h_c_o1

            # Orifice 2 (gap3 between plank2_top and plank3_bottom)
            gap_o2 = plank3_bottom - plank2_top
            if gap_o2 > 0:
                h_c_o2 = coeff_contraction * gap_o2
                M_total += q_orifice2 * q_orifice2 / h_c_o2

            # Weir: critical depth h_c = (q²/g)^(1/3), V_c = q/h_c, M = q²/h_c
            h_c_weir_safe = np.where(q_weir > 1e-9, np.power(q_weir * q_weir / g, 1/3), 1.0)
            M_total += np.divide(
                q_weir * q_weir, h_c_weir_safe,
                out=np.zeros_like(q_weir),
                where=q_weir > 1e-9,
            )

            q_total = q_sluice + q_orifice1 + q_orifice2 + q_weir

            if np.isscalar(h) or (isinstance(h, np.ndarray) and h.ndim == 0):
                return q_total[0], M_total[0]
            return q_total, M_total

        return fn

    def simulate(self):
        def bed_fn(x):
            return -self.incline * x
        
        manning_fn = self._get_mannings_fn()

        if self.barrier:
            barrier_fn = self._get_barrier_fn(self.barrier)
            _, profile = simulate_barrier(self.flow, bed_fn, manning_fn, barrier_fn, self.barrier)
        else:
            _, profile = simulate(self.flow, bed_fn, manning_fn)

        return profile
        
