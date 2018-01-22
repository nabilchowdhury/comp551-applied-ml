import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame([[1, 2],
                   [2, 2],
                   [3, 2],
                   [4, 2],
                   [5, 2],
                   [6, 2],
                   [7, 2],
                   [8, 2],
                   [9, 2],
                   [10, 2],
                   [11, 2]])


print(df[1])
data = np.array(df, dtype=np.float64)


def cross_validation_split(X, n_folds=5, filename="file"):
    N = len(X) // 5
    for i in range(n_folds):
        fold_train1 = X[0:i*N]
        fold_test = X[i*N:(i+1)*N]
        fold_train2 = X[(i+1)*N:]

        df_train = pd.DataFrame(np.concatenate((fold_train1, fold_train2)))
        df_test = pd.DataFrame(fold_test)

        df_train.to_csv('./Datasets/' + filename + '-train' + str(i + 1) + '.csv', header=False, index=False)
        df_test.to_csv('./Datasets/' + filename + '-test' + str(i + 1) + '.csv', header=False, index=False)

cross_validation_split(data)
