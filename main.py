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

        simple_sluice_model = SimpleSluiceModel("simpleSluice")
        sluice_dataframe = simple_sluice_model._create_model_dataframe(lab_data)
        simple_sluice_model.fit(sluice_dataframe)
        simple_sluice_model.write_report(reports_dir)

        simple_weir_model = SimpleWeirModel("simpleWeir")
        sluice_dataframe = simple_weir_model._create_model_dataframe(lab_data)
        simple_weir_model.fit(sluice_dataframe)
        simple_weir_model.write_report(reports_dir)

        simple_orifice_model = SimpleOrificeModel("simpleOrifice")
        orifice_dataframe = simple_orifice_model._create_model_dataframe(lab_data)
        simple_orifice_model.fit(orifice_dataframe)
        simple_orifice_model.write_report(reports_dir)

        #simple_combined_model = SimpleCombinedModel("simpleCombined")

    if args.visualisation:
        pass
    if args.solver:
        test_connection()


if __name__ == "__main__":
    main()
