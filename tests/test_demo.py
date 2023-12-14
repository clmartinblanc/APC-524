from __future__ import annotations
from plotting_assist import *


# making sure I can write a test that successfully imports the function of interest and appropriately passes
def test_add1():
    assert 2 == add1(1)
    assert 2 != add1(4)
