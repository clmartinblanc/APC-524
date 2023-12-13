from __future__ import annotations
from data_classes import data_classes


def test_pass():
    pass


def test_add2():
    assert 4 == data_classes.add2(2)

def test_TableData_instance():
    test_instance = data_classes.TableData(
    "/Users/michaelschroeder/Downloads/test_data.txt", ".txt"
    )
    assert test_instance.get_array() == [['x', 'y', 'z'], [8.6, 5.6, 1.0], [99.3, 77.0, 2.0], [8.01, 44.3, 3.0]]
