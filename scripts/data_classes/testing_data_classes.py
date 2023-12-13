"""
I put test_data.txt in git so everyone can have it.

The code runs using this file stored in your personal computer.

Change the filepath to where it is stored.
"""

print("\n--------------\n")

import numpy as np
import data_classes as data_classes_py
import pandas as pd


test_instance = data_classes_py.TableData("scripts/data_classes/test_data.txt", ".txt")
my_array = test_instance.get_array()

print(my_array)

print(type(test_instance))
