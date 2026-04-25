# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from src.cli import *
from src.analysis import *
from src.visualisation import *
from src.solver import *


def main():
    parser = CustomParser()
    args = parser.parse_args()

    if args.analysis:
        lab_data = read_barrier_data()
        barrier_csv = Path("data/BarrierExperiments.csv")
        reports_dir = Path("exports/reports")

        write_friction_report("frictionValues.csv", reports_dir)

        simple_sluice_model = SimpleSluiceModel("simpleSluice", lab_data)
        simple_sluice_model.fit()
        simple_sluice_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, simple_sluice_model, reports_dir/"simpleSluice.txt")

        simple_weir_model = SimpleWeirModel("simpleWeir", lab_data)
        simple_weir_model.fit()
        simple_weir_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, simple_weir_model, reports_dir/"simpleWeir.txt")

        simple_orifice_model = SimpleOrificeModel("simpleOrifice", lab_data)
        simple_orifice_model.fit()
        simple_orifice_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, simple_orifice_model, reports_dir/"simpleOrifice.txt")

        simple_combined_model = SimpleCombinedModel("simpleCombined", lab_data)
        simple_combined_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, simple_combined_model, reports_dir/"simpleCombined.txt")

        if_combined_model = SimpleIFCombinedModel("simpleIfCombined", lab_data)
        if_combined_model.fit()
        if_combined_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, if_combined_model, reports_dir/"simpleIfCombined.txt")

        advanced_sluice_model = AdvancedSluiceModel("advancedSluice", lab_data)
        advanced_sluice_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, advanced_sluice_model, reports_dir/"advancedSluice.txt")

        advanced_orifice_model = AdvancedOrificeModel("advancedOrifice", lab_data)
        advanced_orifice_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, advanced_orifice_model, reports_dir/"advancedOrifice.txt")

        advanced_weir_model = AdvancedWeirModel("advancedWeir", lab_data)
        advanced_weir_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, advanced_weir_model, reports_dir/"advancedWeir.txt")

        advanced_combined_model = AdvancedCombinedModel("advancedCombined", lab_data)
        advanced_combined_model.write_report(reports_dir)
        run_monte_carlo_analysis(barrier_csv, advanced_combined_model, reports_dir/"advancedCombined.txt")

    if args.visualisation:
        lab_data = read_barrier_data()

        visualisation_1_1(lab_data)

        visualisation_1_2(lab_data)

        visualisation_1_3(lab_data)

        visualisation_1_4()

        visualisation_1_5(lab_data)

    if args.solver:
        test_flume = Flume("", 0.25)
        test_flume.simulate()


if __name__ == "__main__":
    main()
