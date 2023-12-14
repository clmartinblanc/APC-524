import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(data_structure):
    output_files = data_structure.output_files()

    # Initialzie Plot
    fig, ax = plt.subplots()

    for i in output_files:
        data_table = pd.read_csv(data_structure.save_path + "/" + i)
        ax.plot(data_table.y)

        # Customise some display properties
        ax.set_ylabel("y")
        ax.set_title("y vs t")
        ax.set_xlabel("t")

        # Ask Matplotlib to show the plot
        plt.show()
