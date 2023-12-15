import matplotlib.pyplot as plt
import pandas as pd


def plot_all(data_structure):
    output_files = data_structure.output_files()

    # Initialzie Plot
    fig, ax = plt.subplots(1, 2)

    for i in output_files:
        data_table = pd.read_csv(data_structure.save_path + "/" + i + ".csv")
        ax[0].plot(data_table.y, label=i)
        ax[1].plot(data_table.x, label=i)

    # Customise some display properties
    ax[0].set_ylabel("y")
    ax[0].set_title("y vs t")
    ax[0].legend()

    ax[1].set_ylabel("x")
    ax[1].set_title("x vs t")
    ax[1].set_xlabel("t")
    ax[1].legend()

    # Ask Matplotlib to show the plot
    plt.show()
