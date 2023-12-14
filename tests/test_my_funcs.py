from __future__ import annotations
import os
import sys
import types

# have my_funcs, was having issues getting python to find it
current_directory = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(current_directory, ".."))
video_scripts_path = os.path.join(root, "code")
sys.path.append(video_scripts_path)
from my_funcs import *


# verifying that all functions that should be defined are
def test_func_existance():
    assert isinstance(cart_to_wf, types.FunctionType)
    assert isinstance(ene_flux_p, types.FunctionType)
    assert isinstance(ene_flux_v, types.FunctionType)
    assert isinstance(get_amp, types.FunctionType)
    assert isinstance(interp_2d, types.FunctionType)
    assert isinstance(load_bin, types.FunctionType)
    assert isinstance(mom_flux_p, types.FunctionType)
    assert isinstance(mom_flux_p_alt, types.FunctionType)
    assert isinstance(mom_flux_v_alt, types.FunctionType)
    assert isinstance(mom_flux_v, types.FunctionType)
    assert isinstance(phase_partion, types.FunctionType)
    assert isinstance(pol2cart, types.FunctionType)
    assert isinstance(return_file, types.FunctionType)
    assert isinstance(spectrum_integration, types.FunctionType)
    assert isinstance(extract_custar_from_dir, types.FunctionType)
    assert isinstance(extract_direction_from_dir, types.FunctionType)
    assert isinstance(process_directory, types.FunctionType)
    assert isinstance(read_files_to_dfs, types.FunctionType)
    assert isinstance(plot_data_air, types.FunctionType)
    assert isinstance(darken_color, types.FunctionType)
    assert isinstance(plot_data_air_color, types.FunctionType)
    assert isinstance(plot_water_data, types.FunctionType)
    assert isinstance(darken_color, types.FunctionType)
    assert isinstance(plot_data_water_color, types.FunctionType)
    assert isinstance(process_data, types.FunctionType)
    assert isinstance(process_and_plot, types.FunctionType)
    assert isinstance(spectrum_integration, types.FunctionType)
    assert isinstance(exponential_func, types.FunctionType)
