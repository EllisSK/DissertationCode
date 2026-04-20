# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np
import plotly.express as px

from .solver import run_simulation
from typing import Callable

class Flume:
    def __init__(self, barrier_setup: str, set_flow: float):
        self.barrier = barrier_setup
        self.flow = set_flow

    def _get_barrier_fn(self) -> Callable:
        def fn(h, ds):
            if h < 0.4:
                return 0
            else:
                return 0.61 * (2/3) * np.sqrt(2 * 9.81) * ((h-0.4) ** 1.5)

        return fn

    def simulate(self):
        barrier_fn = self._get_barrier_fn()

        results = run_simulation(self.flow, barrier_fn)

        Q = list(results.values())[0]
        h_profile = Q[:, 0]

        fig = px.line(h_profile)
        fig.show()