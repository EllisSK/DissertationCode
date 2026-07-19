# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from pathlib import Path

import pandas as pd

from src.analysis import (
    AdvancedCombinedModel,
    AdvancedOrificeModel,
    AdvancedSluiceModel,
    AdvancedWeirModel,
    SimpleCombinedModel,
    SimpleIFCombinedModel,
    SimpleOrificeModel,
    SimpleSluiceModel,
    SimpleWeirModel,
    read_barrier_data,
    remove_submerged,
    run_friction_monte_carlo_analysis,
    run_monte_carlo_analysis,
    validate_barrier_experiments,
    validate_friction_experiments,
    write_friction_report,
)
from src.cli import CustomParser
from src.solver import reproduce_barrier_experiments, reproduce_friction_experiments
from src.visualisation import (
    generate_paper_figures,
    visualisation_1_1,
    visualisation_1_2,
    visualisation_1_3,
    visualisation_1_5,
    visualisation_1_6,
    visualisation_1_7,
    visualisation_1_8,
)

MODELS = [
    ("simpleSluice", SimpleSluiceModel),
    ("simpleWeir", SimpleWeirModel),
    ("simpleOrifice", SimpleOrificeModel),
    ("simpleCombined", SimpleCombinedModel),
    ("simpleIfCombined", SimpleIFCombinedModel),
    ("advancedSluice", AdvancedSluiceModel),
    ("advancedOrifice", AdvancedOrificeModel),
    ("advancedWeir", AdvancedWeirModel),
    ("advancedCombined", AdvancedCombinedModel),
]


def run_analysis():
    reports_dir = Path("exports/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    lab_data = read_barrier_data()
    raw_lab_data = pd.read_csv(Path("data/BarrierExperiments.csv"))

    datasets = {
        "unfiltered": (lab_data, raw_lab_data),
        "filtered": (remove_submerged(lab_data), remove_submerged(raw_lab_data)),
    }

    write_friction_report("frictionValues.csv", reports_dir)
    run_friction_monte_carlo_analysis()

    for subdir_name, (data, raw_data) in datasets.items():
        report_dir = reports_dir / subdir_name
        report_dir.mkdir(parents=True, exist_ok=True)

        for name, model_class in MODELS:
            model = model_class(name, data)
            model.fit()
            model.write_report(report_dir)
            run_monte_carlo_analysis(raw_data, model, report_dir / f"{name}.txt")


def run_visualisations():
    filtered_lab_data = remove_submerged(read_barrier_data())
    filtered_raw_lab_data = remove_submerged(pd.read_csv(Path("data/BarrierExperiments.csv")))
    fric_data = pd.read_csv(Path("data/ManningsNExperiments.csv"))

    visualisation_1_1(filtered_lab_data)
    visualisation_1_2(filtered_lab_data)
    visualisation_1_3(filtered_lab_data)
    visualisation_1_5(filtered_lab_data)
    visualisation_1_6(filtered_raw_lab_data)
    visualisation_1_7(fric_data)
    visualisation_1_8()


def run_solver():
    reports_dir = Path("exports/reports")

    raw_lab_data = pd.read_csv(Path("data/BarrierExperiments.csv"))
    filtered_raw_lab_data = remove_submerged(raw_lab_data)

    reproduce_friction_experiments()
    reproduce_barrier_experiments(filtered_raw_lab_data)

    validate_friction_experiments(reports_dir)
    validate_barrier_experiments(filtered_raw_lab_data, reports_dir)


def main():
    parser = CustomParser()
    args = parser.parse_args()

    if args.analysis:
        run_analysis()

    if args.visualisation:
        run_visualisations()

    if args.solver:
        run_solver()

    if args.paper:
        generate_paper_figures()


if __name__ == "__main__":
    main()
