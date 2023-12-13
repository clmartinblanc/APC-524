# Code

This part of the repository contains various scripts and files that are essential for our data processing and analysis workflow.

## Postprocessing

This section includes diverse file types, each serving a specific purpose:

- **Data Preparation Scripts**: These scripts are designed to prepare and clean the data for further manipulation, as `Postprocessing.ipynb` . They are crucial in the initial stages of our workflow.

  The 'process_data' is a function crafted for the extraction and processing of simulation data encapsulated within binary files. Operating within a designated working directory, this function is adept at managing outputs generated at distinct simulation time steps. The core data resides in a file named 'global_int.out', which encompasses the time steps 'time' and iteration counts 'istep_c'. An empty series array, 'eta_series, is initialized by the function to retain the processed data. It traverses through each iteration count, retrieving the corresponding binary files that hold the raw output of the simulation. These files undergo reshaping and filtering predicated on specific parameters, such as a defined threshold on the 13th column. Employing the 'numpy' library, a mesh grid is constructed to depict the spatial domain. The data is then interpolated onto this grid using the 'griddata' method. This strategy transforms the unstructured data into a structured form, apt for subsequent analysis and visualization. At each time step, the function assembles a dictionary comprising data, inclusive of average values for particular fields, namely'ux', 'uy', 'uz' (velocity components), 'fv', and 'pressure'. Average calculations are performed across the second axis for these fields to yield a representative mean for each time step. In culmination, these dictionaries are collated into a list and transmuted into a pandas DataFrame. This DataFrame presents a pristine and methodically structured representation of the simulation data, thereby facilitating an efficient pathway for post-processing and thorough analysis.

- **Analysis Notebooks**: Once we have the data in a cleaner and more usable format, you can utilize notebooks like `Spectra.ipynb` and `Graphs.ipynb` for in-depth analysis.

- **Python Utility Files**: The `.py` files contain function definitions that are used across various notebooks. It's beneficial to consolidate more functions in these files. Alternatively, consider restructuring the codebase to use classes and instances for better organization and reusability.


## Data Details


Our data is structured in a binary form, for the different times eta, so the file is named eta_loc_0000t.bin, where t is the time of the simulation that is stored in another variable: global_int.out, where the first column is the time t and the second column is the iteration i.
