import numpy as np

from .basemodel import BaseModel

class SluiceModel(BaseModel):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _equation(self, upstream_depth, gap_size):
        coeff_contraction = np.pi / (np.pi + 2)
        coeff_velocity = 0.99

        coeff_discharge = coeff_contraction * coeff_velocity
        vc_depth = coeff_contraction * gap_size

        head = upstream_depth - vc_depth

        return coeff_discharge * np.sqrt(2 * 9.80665 * head)