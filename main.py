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
        reports_dir = Path("exports/reports")
        unfiltered_dir = reports_dir / "unfiltered"
        filtered_dir = reports_dir / "filtered"

        unfiltered_dir.mkdir(parents=True, exist_ok=True)
        filtered_dir.mkdir(parents=True, exist_ok=True)

        lab_data = read_barrier_data()
        raw_lab_data = pd.read_csv(Path("data/BarrierExperiments.csv"))
        
        filtered_lab_data = remove_submerged(lab_data)
        filtered_raw_lab_data = remove_submerged(raw_lab_data)
        
        write_friction_report("frictionValues.csv", reports_dir)
        run_friction_monte_carlo_analysis()

        simple_sluice_model = SimpleSluiceModel("simpleSluice", lab_data)
        simple_sluice_model.fit()
        simple_sluice_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, simple_sluice_model, unfiltered_dir/"simpleSluice.txt")

        simple_weir_model = SimpleWeirModel("simpleWeir", lab_data)
        simple_weir_model.fit()
        simple_weir_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, simple_weir_model, unfiltered_dir/"simpleWeir.txt")

        simple_orifice_model = SimpleOrificeModel("simpleOrifice", lab_data)
        simple_orifice_model.fit()
        simple_orifice_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, simple_orifice_model, unfiltered_dir/"simpleOrifice.txt")

        simple_combined_model = SimpleCombinedModel("simpleCombined", lab_data)
        simple_combined_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, simple_combined_model, unfiltered_dir/"simpleCombined.txt")

        if_combined_model = SimpleIFCombinedModel("simpleIfCombined", lab_data)
        if_combined_model.fit()
        if_combined_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, if_combined_model, unfiltered_dir/"simpleIfCombined.txt")

        advanced_sluice_model = AdvancedSluiceModel("advancedSluice", lab_data)
        advanced_sluice_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, advanced_sluice_model, unfiltered_dir/"advancedSluice.txt")

        advanced_orifice_model = AdvancedOrificeModel("advancedOrifice", lab_data)
        advanced_orifice_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, advanced_orifice_model, unfiltered_dir/"advancedOrifice.txt")

        advanced_weir_model = AdvancedWeirModel("advancedWeir", lab_data)
        advanced_weir_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, advanced_weir_model, unfiltered_dir/"advancedWeir.txt")

        advanced_combined_model = AdvancedCombinedModel("advancedCombined", lab_data)
        advanced_combined_model.write_report(unfiltered_dir)
        run_monte_carlo_analysis(raw_lab_data, advanced_combined_model, unfiltered_dir/"advancedCombined.txt")

        filtered_simple_sluice_model = SimpleSluiceModel("simpleSluice", filtered_lab_data)
        filtered_simple_sluice_model.fit()
        filtered_simple_sluice_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_simple_sluice_model, filtered_dir/"simpleSluice.txt")

        filtered_simple_weir_model = SimpleWeirModel("simpleWeir", filtered_lab_data)
        filtered_simple_weir_model.fit()
        filtered_simple_weir_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_simple_weir_model, filtered_dir/"simpleWeir.txt")

        filtered_simple_orifice_model = SimpleOrificeModel("simpleOrifice", filtered_lab_data)
        filtered_simple_orifice_model.fit()
        filtered_simple_orifice_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_simple_orifice_model, filtered_dir/"simpleOrifice.txt")

        filtered_simple_combined_model = SimpleCombinedModel("simpleCombined", filtered_lab_data)
        filtered_simple_combined_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_simple_combined_model, filtered_dir/"simpleCombined.txt")

        filtered_if_combined_model = SimpleIFCombinedModel("simpleIfCombined", filtered_lab_data)
        filtered_if_combined_model.fit()
        filtered_if_combined_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_if_combined_model, filtered_dir/"simpleIfCombined.txt")

        filtered_advanced_sluice_model = AdvancedSluiceModel("advancedSluice", filtered_lab_data)
        filtered_advanced_sluice_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_advanced_sluice_model, filtered_dir/"advancedSluice.txt")

        filtered_advanced_orifice_model = AdvancedOrificeModel("advancedOrifice", filtered_lab_data)
        filtered_advanced_orifice_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_advanced_orifice_model, filtered_dir/"advancedOrifice.txt")

        filtered_advanced_weir_model = AdvancedWeirModel("advancedWeir", filtered_lab_data)
        filtered_advanced_weir_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_advanced_weir_model, filtered_dir/"advancedWeir.txt")

        filtered_advanced_combined_model = AdvancedCombinedModel("advancedCombined", filtered_lab_data)
        filtered_advanced_combined_model.write_report(filtered_dir)
        run_monte_carlo_analysis(filtered_raw_lab_data, filtered_advanced_combined_model, filtered_dir/"advancedCombined.txt")

    if args.visualisation:
        lab_data = read_barrier_data()
        filtered_lab_data = remove_submerged(lab_data)

        raw_lab_data = pd.read_csv(Path("data/BarrierExperiments.csv"))
        filtered_raw_lab_data = remove_submerged(raw_lab_data)
        
        fric_data = pd.read_csv(Path("data/ManningsNExperiments.csv"))

        visualisation_1_1(filtered_lab_data)

        visualisation_1_2(filtered_lab_data)

        visualisation_1_3(filtered_lab_data)

        visualisation_1_5(filtered_lab_data)

        visualisation_1_6(filtered_raw_lab_data)

        visualisation_1_7(fric_data)

        visualisation_1_8()

    if args.solver:
        reports_dir = Path("exports/reports")

        raw_lab_data = pd.read_csv(Path("data/BarrierExperiments.csv"))
        filtered_raw_lab_data = remove_submerged(raw_lab_data)

        reproduce_friction_experiments()

        reproduce_barrier_experiments(filtered_raw_lab_data)

        validate_friction_experiments(reports_dir)

        validate_barrier_experiments(filtered_raw_lab_data, reports_dir)


if __name__ == "__main__":
    main()
