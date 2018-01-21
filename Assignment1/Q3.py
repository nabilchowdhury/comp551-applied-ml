import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
3) Real-life dataset
'''

df = pd.read_csv('./Datasets/communities.csv', header=None)
df.replace('?', np.NaN, inplace=True)
# obj_cols = [i for i in range(128) if df.dtypes.iloc[i] == 'object']
# print(df[obj_cols].head())

'''
Part 1) Fill in missing values 
'''
print(df.mean())
df.apply(lambda col: col.fillna(col.mean()), axis=0)
print(df.head())

print(df.isna().any())