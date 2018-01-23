import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alphas = [(pow(10, -i), 5 * pow(10, -i)) for i in range(7)]
alphas = [item for sublist in alphas for item in sublist]

print(alphas)