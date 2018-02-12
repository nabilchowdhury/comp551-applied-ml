import numpy as np
import pandas as pd

generators = [
    np.random.multivariate_normal([1, 1], [[5, 1], [1, 5]]),
    np.random.multivariate_normal([0, 0], [[5, 1], [1, 5]]),
    np.random.multivariate_normal([-1, -1], [[5, 1], [1, 5]])]

draw = np.random.choice([0, 1, 2], 100, p=[0.7, 0.2, 0.1])

print([generators[i] for i in draw])
