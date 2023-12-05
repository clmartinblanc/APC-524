"""
I put test_data.txt in git so everyone can have it. 

The code runs using this file stored in your personal computer. 

Change the filepath to where it is stored.
"""

print("\n--------------\n")

import numpy as np
import data_classes
import pandas as pd


test_instance = data_classes.TableData("/Users/michaelschroeder/Downloads/test_data.txt",".txt")
my_array = test_instance.getArray()

print(my_array)