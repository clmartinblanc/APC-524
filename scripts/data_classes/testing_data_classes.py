"""
I put test_data.csv in git so everyone can have it.

The code runs using this file stored in your personal computer.

Change the filepath to where it is stored.
"""

print("\n--------------\n")

import numpy as np
import data_classes as data_classes
import pandas as pd


test_instance = data_classes.TableData("scripts/data_classes/test_data.csv", ".csv")
my_array = test_instance.get_array()
print(my_array)
print(type(test_instance))

csv_instance = data_classes.TableData("scripts/video_demo/output/Data_RunA.csv", ".csv")
ar = csv_instance.get_array()
print(ar)
