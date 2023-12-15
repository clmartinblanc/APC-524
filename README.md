# APC-524 Final Project

[![.github/workflows/ci.yml](https://github.com/clmartinblanc/APC-524/actions/workflows/ci.yml/badge.svg)](https://github.com/clmartinblanc/APC-524/actions/workflows/ci.yml)

## Overview

This code is intended to streamline the process of analyzing large sets of data commonly produced by experimental or numerical fluid mechanics research. Generally, it can be applied to any large dataset consisting of many similar data files which all need to be analyzed with the same scripts.

## Class Structure

The core of this project is a class structure which stores the data path, output file path, and data file extension. Using this structure, analysis and plotting scripts can be run on all data using a single line of code per script.

The data class definitions are found in `scripts/data_classes/data_classes.py` and contains an abstract `Data` class and two sub-classes: `VideoData` and `TableData`. New subclasses can be added here to handle other datatypes that may not fit into these categories.

## Examples

This repo contains two examples, applying the the class structure to video and binary datasets.

### Video Data
 The demo of analyzing video data is found in `scripts/video_demo/video_demo.ipynb`. The dataset consists of a toy problem of tracking the center of a bouncing ball. `scripts/video_demo/demo` contains data from three trials in the form of serialized images. `scripts/video_demo/video_scripts` contains `edge_detection.py`, which uses open-cv to output the coordinates and radius of the circle for each frame to a .csv file stored in `scripts/video_demo/output`. It also contains `video_plot.py` which plots the x and y coordinates vs time for all data.

### Binary Data
#### Postprocessing

This section includes diverse file types, each serving a specific purpose:

- **Data Preparation Scripts**: These scripts are designed to prepare and clean the data for further manipulation, as `Postprocessing.ipynb` . They are crucial in the initial stages of our workflow.

  The `process_data` is a function crafted for the extraction and processing of simulation data encapsulated within binary files. Operating within a designated working directory, this function is adept at managing outputs generated at distinct simulation time steps. The core data resides in a file named `global_int.out`, which encompasses the time steps `time` and iteration counts `istep_c`. An empty series array, `eta_series`, is initialized by the function to retain the processed data. It traverses through each iteration count, retrieving the corresponding binary files that hold the raw output of the simulation. These files undergo reshaping and filtering predicated on specific parameters, such as a defined threshold on the 13th column. Employing the `numpy` library, a mesh grid is constructed to depict the spatial domain. The data is then interpolated onto this grid using the `griddata` method. This strategy transforms the unstructured data into a structured form, apt for subsequent analysis and visualization. At each time step, the function assembles a dictionary comprising data, inclusive of average values for particular fields, namely `ux`, `uy`, `uz` (velocity components), `fv`, and `pressure`. Average calculations are performed across the second axis for these fields to yield a representative mean for each time step. In culmination, these dictionaries are collated into a list and transmuted into a pandas DataFrame. This DataFrame presents a pristine and methodically structured representation of the simulation data, thereby facilitating an efficient pathway for post-processing and thorough analysis.

- **Analysis Notebooks**: Once we have the data in a cleaner and more usable format, you can utilize notebooks like `Spectra.ipynb` and `Graphs.ipynb` for in-depth analysis.

- **Python Utility Files**: The `.py` files contain function definitions that are used across various notebooks. It's beneficial to consolidate more functions in these files. Alternatively, consider restructuring the codebase to use classes and instances for better organization and reusability.


#### Data Details


Our data is structured in a binary form, for the different times `eta`, so the file is named `eta_loc_0000t.bin`, where `t` is the time of the simulation that is stored in another variable: `global_int.out`, where the first column is the time `t` and the second column is the iteration `i`.


