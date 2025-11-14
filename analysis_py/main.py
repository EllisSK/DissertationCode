import sys
import pathlib
import numpy as np
from analysis.data_processing import read_lab_data, create_sluice_dataframe
from visualisation.plots import create_sluice_cd_ha_plot, add_function_to_plot


def main():
    if len(sys.argv) == 1:
        print("No path to data given! Please provide the path to the lab data.")
        return
    
    df = create_sluice_dataframe(read_lab_data(pathlib.Path(sys.argv[1])))

    fig = create_sluice_cd_ha_plot(df)

    fig.show()

if __name__ == "__main__":
    main()