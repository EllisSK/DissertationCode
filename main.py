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
        sluice_data = create_sluice_dataframe(lab_data)
        create_model_report(
            sluice_equation,
            sluice_data,
            "sluice_cd",
            "Flow (m3/s)",
            "Total Head (m)",
            "Sluice Area (m2)",
        )
    if args.visualisation:
        lab_data = read_lab_data()
        sluice_data = create_sluice_dataframe(lab_data)
        fig = create_flow_head_plot(sluice_data)
        fig = add_function_to_plot(fig, sluice_equation, (0.15, 0.6), 0.01, "150 (Model)", 0.15, 0.611)
        fig = add_function_to_plot(fig, sluice_equation, (0.1, 0.6), 0.01, "100 (Model)", 0.1, 0.611)
        fig = add_function_to_plot(fig, sluice_equation, (0.05, 0.6), 0.01, "50 (Model)", 0.05, 0.611)
        save_figure(fig, "sluice_flow_head", "sluice")
    if args.solver:
        test_connection()


if __name__ == "__main__":
    main()
