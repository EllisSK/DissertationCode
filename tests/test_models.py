# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    AdvancedOrificeModel,
    AdvancedSluiceModel,
    AdvancedWeirModel,
    SimpleCombinedModel,
    SimpleOrificeModel,
    SimpleSluiceModel,
    SimpleWeirModel,
)


def make_lab_data(
    operation_mode: str, barrier_setup: str, flows_depths: list[tuple[float, float]]
) -> pd.DataFrame:
    """Build a minimal lab-data frame as produced by read_barrier_data()."""
    return pd.DataFrame(
        {
            "Barrier Setup": barrier_setup,
            "Operation Mode": operation_mode,
            "Set Flow (l/s)": [f for f, _ in flows_depths],
            "Mean Upstream Depth (mm)": [d for _, d in flows_depths],
        }
    )


def synthetic_sluice_data(
    coeff: float, gap_mm: int, depths_mm: list[float]
) -> pd.DataFrame:
    """Generate flows that satisfy the simple sluice equation exactly."""
    gap_m = gap_mm / 1000
    flows_ls = [coeff * gap_m * np.sqrt(d / 1000) * 1000 for d in depths_mm]
    return make_lab_data("Sluice", f"{gap_mm}-100-50", list(zip(flows_ls, depths_mm)))


class TestSimpleSluiceModel:
    def test_equation(self):
        model = SimpleSluiceModel(
            "test", make_lab_data("Sluice", "100-100-50", [(10.0, 250.0)])
        )
        # Q = C * a * sqrt(h)
        assert model._equation((0.25, 0.1), 2.0) == pytest.approx(
            2.0 * 0.1 * np.sqrt(0.25)
        )

    def test_fit_recovers_known_coefficient(self):
        model = SimpleSluiceModel(
            "test", synthetic_sluice_data(2.1, 100, [150.0, 250.0, 350.0, 450.0])
        )
        model.fit()
        assert model.optimal == pytest.approx(2.1, rel=1e-6)

    def test_predict_before_fit_raises(self):
        model = SimpleSluiceModel(
            "test", synthetic_sluice_data(2.1, 100, [150.0, 250.0])
        )
        with pytest.raises(Exception, match="hasn't been fit"):
            model.predict((0.25, 0.1))

    def test_perfect_fit_metrics(self):
        model = SimpleSluiceModel(
            "test", synthetic_sluice_data(2.1, 100, [150.0, 250.0, 350.0])
        )
        model.fit()
        rmse, mae, bias, var, corr, kge, r2 = model._calculate_objective_functions(
            model.df
        )
        assert rmse == pytest.approx(0.0, abs=1e-12)
        assert mae == pytest.approx(0.0, abs=1e-12)
        assert bias == pytest.approx(0.0, abs=1e-12)
        assert var == pytest.approx(1.0)
        assert corr == pytest.approx(1.0)
        assert kge == pytest.approx(1.0)
        assert r2 == pytest.approx(1.0)


class TestSimpleWeirModel:
    def test_equation(self):
        model = SimpleWeirModel("test", make_lab_data("Weir", "0-0-0", [(10.0, 450.0)]))
        # Q = C * h^1.5
        assert model._equation(0.09, 1.8) == pytest.approx(1.8 * 0.09**1.5)

    def test_weir_height_from_setup(self):
        # 0-0-0: all three planks stacked -> crest at 0.4 m
        model = SimpleWeirModel("test", make_lab_data("Weir", "0-0-0", [(10.0, 450.0)]))
        assert model.df["Weir Height (m)"].iloc[0] == pytest.approx(0.4)
        assert model.df["Head on Weir (m)"].iloc[0] == pytest.approx(0.05)

    def test_submerged_crest_rows_dropped(self):
        # Upstream depth below the crest gives no head on the weir -> row removed
        model = SimpleWeirModel("test", make_lab_data("Weir", "0-0-0", [(10.0, 350.0)]))
        assert model.df.empty


class TestSimpleOrificeModel:
    def test_equation(self):
        model = SimpleOrificeModel(
            "test", make_lab_data("Orifice", "0-100-0", [(10.0, 350.0)])
        )
        # Q = C * (h_bottom^1.5 - h_top^1.5)
        assert model._equation((0.15, 0.05), 2.0) == pytest.approx(
            2.0 * (0.15**1.5 - 0.05**1.5)
        )

    def test_orifice_geometry_middle_gap(self):
        # 0-100-0: orifice is the 100 mm gap above the bottom 0.2 m plank
        model = SimpleOrificeModel(
            "test", make_lab_data("Orifice", "0-100-0", [(10.0, 350.0)])
        )
        row = model.df.iloc[0]
        assert row["Orifice Bottom Height (m)"] == pytest.approx(0.2)
        assert row["Orifice Top Height (m)"] == pytest.approx(0.3)

    def test_orifice_geometry_top_gap(self):
        # 0-0-50: orifice is the 50 mm gap above the 0.3 m double plank stack
        model = SimpleOrificeModel(
            "test", make_lab_data("Orifice", "0-0-50", [(10.0, 400.0)])
        )
        row = model.df.iloc[0]
        assert row["Orifice Bottom Height (m)"] == pytest.approx(0.3)
        assert row["Orifice Top Height (m)"] == pytest.approx(0.35)


class TestAdvancedModels:
    def test_sluice_equation_analytic(self):
        model = AdvancedSluiceModel(
            "test", make_lab_data("Sluice", "100-100-50", [(10.0, 250.0)])
        )
        cc = np.pi / (np.pi + 2)
        cd = cc * 0.98
        head = 0.25 - cc * 0.1
        assert model._equation((0.25, 0.1)) == pytest.approx(
            cd * 0.1 * np.sqrt(2 * 9.80665 * head)
        )

    def test_weir_equation_analytic(self):
        model = AdvancedWeirModel(
            "test", make_lab_data("Weir", "0-0-0", [(10.0, 450.0)])
        )
        cd = (np.pi / (np.pi + 2)) * 0.98
        assert model._equation((0.05, 0.4)) == pytest.approx(
            (2 / 3) * cd * np.sqrt(2 * 9.80665) * 0.05**1.5
        )

    def test_orifice_equation_analytic(self):
        model = AdvancedOrificeModel(
            "test", make_lab_data("Orifice", "0-100-0", [(10.0, 350.0)])
        )
        cd = (np.pi / (np.pi + 2)) * 0.98
        expected = (2 / 3) * cd * np.sqrt(2 * 9.80665) * (0.15**1.5 - 0.05**1.5)
        assert model._equation((0.15, 0.05)) == pytest.approx(expected)


def combined_lab_data() -> pd.DataFrame:
    """Lab data covering all three operation modes, as the combined models fit a sub-model on each."""
    return pd.concat(
        [
            make_lab_data(
                "Sluice", "100-100-50", [(8.0, 150.0), (10.0, 250.0), (12.0, 300.0)]
            ),
            make_lab_data(
                "Weir", "0-0-0", [(6.0, 420.0), (10.0, 450.0), (14.0, 480.0)]
            ),
            make_lab_data(
                "Orifice", "0-100-0", [(7.0, 320.0), (9.0, 350.0), (11.0, 380.0)]
            ),
        ],
        ignore_index=True,
    )


class TestCombinedModelParams:
    def test_flow_regime_partitioning(self):
        model = SimpleCombinedModel("test", combined_lab_data())

        # 100-100-50 with 0.2/0.1/0.1 m planks: plank interfaces at
        # 0.1-0.3 (plank), 0.3-0.4 (gap), 0.4-0.5 (plank), 0.5-0.55 (gap), 0.55-0.65 (plank)
        # Depth below first plank top -> sluice flow only
        params = model._params_from_setup("100-100-50", 0.25)
        assert params == pytest.approx((0.25, 0.1, 0, 0, 0, 0, 0))

        # Depth above everything -> sluice + both orifices + weir
        params = model._params_from_setup("100-100-50", 0.7)
        h_sluice, gap, hb1, ht1, hb2, ht2, h_weir = params
        assert (h_sluice, gap) == pytest.approx((0.7, 0.1))
        assert hb1 == pytest.approx(0.7 - 0.3)
        assert ht1 == pytest.approx(0.7 - 0.4)
        assert hb2 == pytest.approx(0.7 - 0.5)
        assert ht2 == pytest.approx(0.7 - 0.55)
        assert h_weir == pytest.approx(0.7 - 0.65)

    def test_no_bottom_gap_gives_no_sluice_flow(self):
        model = SimpleCombinedModel("test", combined_lab_data())
        params = model._params_from_setup("0-100-50", 0.6)
        assert params[0] == 0
        assert params[1] == 0


class TestObjectiveFunctions:
    def test_metrics_known_values(self):
        from src.analysis import objective

        observed = pd.Series([1.0, 2.0, 3.0, 4.0])
        predicted = pd.Series([1.5, 2.5, 3.5, 4.5])

        assert objective.rmse(observed, predicted) == pytest.approx(0.5)
        assert objective.mae(observed, predicted) == pytest.approx(0.5)
        assert objective.bias(observed, predicted) == pytest.approx(0.5)
        assert objective.variability(observed, predicted) == pytest.approx(1.0)
        assert objective.correlation(observed, predicted) == pytest.approx(1.0)
        assert objective.r2(observed, predicted) == pytest.approx(1 - 4 * 0.25 / 5.0)
