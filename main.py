# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from src.cli import *
from src.analysis import *
from src.visualisation import *


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
        save_figure(fig, "sluice_flow_head", "sluice")


if __name__ == "__main__":
    main()
