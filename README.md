# APC-524 Final Project

[![.github/workflows/ci.yml](https://github.com/clmartinblanc/APC-524/actions/workflows/ci.yml/badge.svg)](https://github.com/clmartinblanc/APC-524/actions/workflows/ci.yml)

## Overview

This code is intended to streamline the process of analyzing large sets of data commonly produced by experimental or numerical fluid mechanics research. Generally, it can be applied to any large dataset consisting of many similar data files which all need to be analyzed with the same scripts.

## Class Structure

The core of this project is a class structure which stores the data path, output file path, and data file extension. Using this structure, analysis and plotting scripts can be run on all data using a single line of code per script.

The data class definitions are found in `scripts/data_classes/data_classes.py` and contains an abstract `Data` class and two sub-classes: `VideoData` and `TableData`. New subclasses can be added here to handle other datatypes that may not fit into these categories.

## Examples

This repo contains two examples, applying the the class structure to video and binary datasets.

-**Video Data**: The demo of analyzing video data is found in `scripts/video_demo/video_demo.ipynb`. The dataset consists of a toy problem of tracking the center of a bouncing ball. `scripts/video_demo/demo` contains data from three trials in the form of serialized images. `scripts/video_demo/video_scripts` contains `edge_detection.py`, which uses open-cv to output the coordinates and radius of the circle for each frame to a .csv file stored in `scripts/video_demo/output`. It also contains `video_plot.py` which plots the x and y coordinates vs time for all data.

-**Binary Data**: FILL THIS IN!!


