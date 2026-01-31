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
        simple_sluice_model = SimpleSluiceModel("simpleSluice")
        sluice_dataframe = simple_sluice_model._create_model_dataframe(lab_data)
        simple_sluice_model.fit(sluice_dataframe)
        simple_sluice_model.write_report(Path("exports/reports"))
    if args.visualisation:
        pass
    if args.solver:
        test_connection()


if __name__ == "__main__":
    main()
