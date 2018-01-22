import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
3) Real-life dataset
'''

df = pd.read_csv('./Datasets/communities.csv', header=None)
df.drop([i for i in range(5)], axis=1, inplace=True) # These columns are not predictive according to the dataset
df.columns = [i for i in range(0, df.shape[1])] # Rename columns
df = df.replace('?', np.NaN).astype(np.float64)
# obj_cols = [i for i in range(128) if df.dtypes.iloc[i] == 'object']
# print(df[obj_cols].head())

'''
Part 1) Fill in missing values 
'''
df.fillna(df.mean(), inplace=True)


N_examples = df.shape[0]
M_rows = df.shape[1]

data = np.array(df, dtype=np.float64)
print(N_examples)
# X_all = np.array(df.drop([M_rows - 1], axis=1), dtype=np.float64)
# y_all = np.array(df[M_rows - 1], dtype=np.float64)


'''
Fit data and report 5-fold cross-validation error
'''
# Normal equation with regularization (lambda_reg=0 means no regularization)
def fit(X, y, lambda_reg=0):
    identity = np.identity(X.shape[1])
    # identity[0, 0] = 0 # We do not penalize the bias term

    X_square = np.matmul(np.transpose(X), X) + lambda_reg * identity
    X_square_inverse = np.linalg.pinv(X_square)
    weights = np.matmul(np.matmul(X_square_inverse, np.transpose(X)), y)

    return weights


def mean_square_error(X, y, W):
    y_hat = np.matmul(X, W)
    mean_square_err = np.sum(np.square(y - y_hat)) / len(y)

    return mean_square_err


def cross_validation_split(X, n_folds=5, filename="file", write_to_csv=False):
    N = len(X) // 5
    pairs = []
    for i in range(n_folds):
        fold_train1 = X[0:i*N]
        fold_test = X[i*N:(i+1)*N]
        fold_train2 = X[(i+1)*N:]

        df_train = pd.DataFrame(np.concatenate((fold_train1, fold_train2)))
        df_test = pd.DataFrame(fold_test)

        if write_to_csv:
            df_train.to_csv('./Datasets/' + filename + '-train' + str(i + 1) + '.csv', header=False, index=False)
            df_test.to_csv('./Datasets/' + filename + '-test' + str(i + 1) + '.csv', header=False, index=False)

        pairs.append({'train': df_train, 'test': df_test})

    return pairs


# Generate the files for 5-fold cross-validation
splits = cross_validation_split(data, 5, 'CandC', False)

MSEs = []
all_weights = []
for pair in splits:
    X_train = pair['train'].drop([M_rows - 1], axis=1)
    y_train = pair['train'][M_rows-1]
    X_test = pair['test'].drop([M_rows - 1], axis=1)
    y_test = pair['test'][M_rows - 1]

    weights = fit(X_train, y_train)
    MSEs.append(mean_square_error(X_test, y_test, weights))
    all_weights.append(weights)

print(np.average(MSEs))
