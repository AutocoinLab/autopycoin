"""
============================
Plotting Stack
============================

An example plot of :class:`autopycoin.models.Stack`
"""

import numpy as np
from matplotlib import pyplot as plt

X = [0, 0]
y = [0, 1]

plt.plot(X, y)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim([-0.5, 1.5])

plt.show()
