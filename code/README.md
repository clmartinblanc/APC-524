# Code

This part of the repository contains various scripts and files that are essential for our data processing and analysis workflow.

## Postprocessing

This section includes diverse file types, each serving a specific purpose:

- **Data Preparation Scripts**: These scripts are designed to prepare and clean the data for further manipulation. They are crucial in the initial stages of our workflow.

- **Analysis Notebooks**: Once we have the data in a cleaner and more usable format, you can utilize notebooks like `Spectra.ipynb` and `Graphs.ipynb` for in-depth analysis.

- **Python Utility Files**: The `.py` files contain function definitions that are used across various notebooks. It's beneficial to consolidate more functions in these files. Alternatively, consider restructuring the codebase to use classes and instances for better organization and reusability.


## Data Details


Our data is structured in a binary form, for the different times eta, so the file is named eta_loc_0000t.bin, where t is the time of the simulation that is stored in another variable: global_int.out, where the first column is the time t and the second column is the iteration i. 
