from __future__ import annotations
import os
import sys
import types
import numpy as np


# have my_funcs, was having issues getting python to find it
current_directory = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(current_directory, ".."))
video_scripts_path = os.path.join(root, "code")
sys.path.append(video_scripts_path)
from my_funcs import *


# verifying that all functions that should be defined are
def test_func_existance():
    assert isinstance(CoordinateConverter.cart_to_wf, types.FunctionType)
    assert isinstance(Utilities.ene_flux_p, types.FunctionType)
    assert isinstance(Utilities.ene_flux_v, types.FunctionType)
    assert isinstance(Utilities.get_amp, types.FunctionType)
    assert isinstance(CoordinateConverter.interp_2d, types.FunctionType)
    assert isinstance(FileHandler.load_bin, types.FunctionType)
    assert isinstance(Utilities.mom_flux_p, types.FunctionType)
    assert isinstance(Utilities.mom_flux_p_alt, types.FunctionType)
    assert isinstance(Utilities.mom_flux_v_alt, types.FunctionType)
    assert isinstance(Utilities.mom_flux_v, types.FunctionType)
    assert isinstance(phase_partion, types.FunctionType)
    assert isinstance(CoordinateConverter.pol2cart, types.FunctionType)
    assert isinstance(FileHandler.return_file, types.FunctionType)
    assert isinstance(SpectrumAnalyzer.spectrum_integration_2d, types.FunctionType)
    assert isinstance(SpectrumAnalyzer.spectrum_integration_3d, types.FunctionType)
    assert isinstance(SpectrumAnalyzer.spectrum_integration, types.FunctionType)
    assert isinstance(DataProcessor.extract_custar_from_dir, types.FunctionType)
    assert isinstance(DataProcessor.extract_direction_from_dir, types.FunctionType)
    assert isinstance(DataProcessor.process_directory, types.FunctionType)
    assert isinstance(DataProcessor.read_files_to_dfs, types.FunctionType)
    assert isinstance(DataPlotter.plot_data, types.FunctionType)
    assert isinstance(DataPlotter.darken_color, types.FunctionType)
    assert isinstance(DataPlotter.plot_data_color, types.FunctionType)
    assert isinstance(DataProcessor.process_data, types.FunctionType)
    assert isinstance(DataPlotter.process_and_plot, types.FunctionType)
    assert isinstance(exponential_func, types.FunctionType)


# testing that pol2cart works correctly
def test_pol2cart():
    assert CoordinateConverter.pol2cart(1, 0) == (1, 0)
    assert CoordinateConverter.pol2cart(0, 0) == (0, 0)
    assert CoordinateConverter.pol2cart(0, 1) == (0, 0)
    x, y = CoordinateConverter.pol2cart(1, np.pi)
    assert round(x, 5) == -1
    assert round(y, 5) == 0
