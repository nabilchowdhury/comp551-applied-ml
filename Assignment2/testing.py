import numpy as np
import pandas as pd

a = [(1,2), (0,2.1), (1,2.2), (1,3), (0,4)]
print(sum(x[1] for x in a))