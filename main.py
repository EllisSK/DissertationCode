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
        lab_data = read_lab_data()
        reports_dir =Path("exports/reports")

        simple_sluice_model = SimpleSluiceModel("simpleSluice", lab_data)
        simple_sluice_model.fit()
        simple_sluice_model.write_report(reports_dir)

        simple_weir_model = SimpleWeirModel("simpleWeir", lab_data)
        simple_weir_model.fit()
        simple_weir_model.write_report(reports_dir)

        simple_orifice_model = SimpleOrificeModel("simpleOrifice", lab_data)
        simple_orifice_model.fit()
        simple_orifice_model.write_report(reports_dir)

        simple_combined_model = SimpleCombinedModel("simpleCombined", lab_data)

    if args.visualisation:
        lab_data = read_lab_data()
        simple_combined_model = SimpleCombinedModel("simpleCombined", lab_data)

        #setups = simple_combined_model.df["Barrier Setup"].unique()
        setups = ["100-100-50"]

        us_depth_fig = create_flow_us_depth_plot(simple_combined_model.df, setups)
        for setup in setups:
            add_function_to_plot(us_depth_fig, simple_combined_model.plotting_function, (0.1, 0.7), 0.001, f"{setup} Model", setup)
        save_figure(us_depth_fig, "CombinedModelTest", "SimpleCombinedModel")
    if args.solver:
        test_connection()


if __name__ == "__main__":
    main()
