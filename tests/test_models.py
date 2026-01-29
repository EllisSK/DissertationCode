# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import pytest
from src.analysis.models import *

def test_sluice_model():
    flow = 0.01
    head = 0.00137021797
    area = 0.1
    Cd = 0.61

    result = sluice_equation(head, area, Cd)

    assert result == pytest.approx(flow)
