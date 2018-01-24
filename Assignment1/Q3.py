import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
3) Real-life dataset
'''

df = pd.read_csv('./Datasets/communities.csv', header=None)
df[0] = 1
df.drop([i for i in range(1, 5)], axis=1, inplace=True) # These columns are not predictive according to the dataset
df.columns = [i for i in range(df.shape[1])] # Rename columns
df = df.replace('?', np.NaN).astype(np.float64)

'''
Part 1) Fill in missing values 
'''
df.fillna(df.mean(), inplace=True)
# print(df.columns[df.isnull().any()].tolist())

N_examples = df.shape[0]
M_cols = df.shape[1]

data = np.array(df, dtype=np.float64)


'''
Part 2) Fit data and report 5-fold cross-validation error
'''
# Normal equation with regularization (lambda_reg=0 means no regularization)
def fit(X, y, lambda_reg=0):
    identity = np.identity(X.shape[1])
    # identity[0, 0] = 0 # We do not penalize the bias term

    X_square = np.matmul(np.transpose(X), X) + lambda_reg * identity
    X_square_inverse = np.linalg.pinv(X_square)
    weights = np.matmul(np.matmul(X_square_inverse, np.transpose(X)), y)

    return weights

# Gradient descent
def gradient_descent(X, y, lambda_reg=0, alpha=1e-2, epochs=5000, weights=None):
    '''
    Implementation of vectorized gradient descent with L2 regularization support
    :param X: The input matrix N x M, where each row is an example
    :param y: The output, N x 1
    :param lambda_reg: Regularization hyperparamter (0 means no regularization)
    :param alpha: Learning rate
    :param epochs: Number of cycles over training data
    :param weights: Initial weights can be supplied if desired
    :return: The optimal weights after gradient descent
    '''
    weights = weights if weights is not None else np.random.uniform(high=10, size=[M_cols - 1])
    N = len(X)
    for epoch in range(epochs):
        weights = weights - alpha / N * ( np.matmul(np.transpose(X), np.matmul(X, weights) - y) + lambda_reg * weights)

    return weights

def mean_square_error(X, y, W):
    y_hat = np.matmul(X, W)
    mean_square_err = np.sum(np.square(y - y_hat)) / len(y)

    return mean_square_err


def cross_validation_split(X, n_folds=5, filename="file", write_to_csv=False):
    N = len(X) // 5
    pairs = []
    for i in range(n_folds):
        fold_train1 = X[0:i * N]
        if i < n_folds - 1:
            fold_test = X[i*N:(i+1)*N]
            fold_train2 = X[(i+1)*N:]
        else:
            fold_test = X[i*N:]
            fold_train2 = X[N:N]

        df_train = pd.DataFrame(np.concatenate((fold_train1, fold_train2)))
        df_test = pd.DataFrame(fold_test)

        if write_to_csv:
            df_train.to_csv('./Datasets/' + filename + '-train' + str(i + 1) + '.csv', header=False, index=False)
            df_test.to_csv('./Datasets/' + filename + '-test' + str(i + 1) + '.csv', header=False, index=False)

        pairs.append({'train': df_train, 'test': df_test})

    return pairs


# Generate the files for 5-fold cross-validation
splits = cross_validation_split(data, 5, 'CandC', False)
LAMBDA_REG = 1e-5
MSEs_closed_form = []
all_weights = []
for pair in splits:
    X_train = pair['train'].drop([M_cols - 1], axis=1)
    y_train = pair['train'][M_cols-1]
    X_test = pair['test'].drop([M_cols - 1], axis=1)
    y_test = pair['test'][M_cols - 1]

    weights = fit(X_train, y_train, lambda_reg=LAMBDA_REG)
    MSEs_closed_form.append(mean_square_error(X_test, y_test, weights))
    all_weights.append(weights)


MSEs_gd = []
all_weights_gd = []
weights = np.random.uniform(high=10., size=[M_cols - 1])
for pair in splits:
    X_train = pair['train'].drop([M_cols - 1], axis=1)
    y_train = pair['train'][M_cols - 1]
    X_test = pair['test'].drop([M_cols - 1], axis=1)
    y_test = pair['test'][M_cols - 1]

    weights_gd = gradient_descent(np.array(X_train), np.array(y_train), epochs=20000, weights=np.copy(weights), lambda_reg=LAMBDA_REG)
    MSEs_gd.append(mean_square_error(X_test, y_test, weights_gd))
    all_weights_gd.append(weights_gd)


print('least squares:', np.average(MSEs_closed_form))
print('gradient descent:', np.average(MSEs_gd))

# print(all_weights[0])
# print(all_weights_gd[0])

'''
Part 3) Ridge-regression: Using only least-squares closed form solution for this part.
'''
INCREMENTS = 1000
MSE_vs_lambda = []
weights_for_lambda = []

for i in range(INCREMENTS):

    cur_mse = 0
    weights_for_folds = []
    for pair in splits:
        X_train = pair['train'].drop([M_cols - 1], axis=1)
        y_train = pair['train'][M_cols - 1]
        X_test = pair['test'].drop([M_cols - 1], axis=1)
        y_test = pair['test'][M_cols - 1]

        weights = fit(X_train, y_train, 1 / INCREMENTS * i)
        cur_mse += mean_square_error(X_test, y_test, weights)
        weights_for_folds.append(weights)
    MSE_vs_lambda.append(cur_mse / len(splits))
    weights_for_lambda.append(weights_for_folds)

print(MSE_vs_lambda[:10])

# plt.figure(1)
# plt.plot(np.arange(0, 1, 1/INCREMENTS), MSE_vs_lambda)
#
# plt.show()
#
# Use lasso for feature selection
