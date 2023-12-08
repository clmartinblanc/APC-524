"""
Reorganizing the plotting stuff into a library of plotting tools that we can use throughout our scripts
"""

import numpy as np
import matplotlib.pyplot as plt

def vec_field_example():
    # creating mesh
    x, y = np.meshgrid(np.linspace(-4, 4, 10), np.linspace(-4, 4, 10))

    # velocity vectors from the MAE 551 midterm Q1
    a = 1
    b = 1
    t = 1
    w = 1
    u = (a + b * np.sin(w * t)) * x
    v = -(a + b * np.sin(w * t)) * y

    plt.quiver(x, y, u, v)
    plt.axis("scaled")

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.title("Velocity Field")
    plt.xlabel("u")
    plt.ylabel("v")

    plt.grid()
    plt.show()
