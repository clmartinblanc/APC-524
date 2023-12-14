from __future__ import annotations
from data_classes import data_classes


# testing appropriate functionality of TableData class
def test_TableData_instance():
    test_instance = data_classes.TableData("scripts/data_classes/test_data.csv", ".csv")

    # verifying properly initiated
    assert test_instance.data_path == "scripts/data_classes/test_data.csv"
    assert test_instance.extension == ".csv"

    # verifying it can correctly access and convert table data
    assert test_instance.get_array() == [
        ["x", "y", "z"],
        [8.6, 5.6, 1.0],
        [99.3, 77.0, 2.0],
        [8.01, 44.3, 3.0],
    ]
