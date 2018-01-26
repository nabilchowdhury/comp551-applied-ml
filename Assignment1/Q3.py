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
df = df.sample(frac=1) # This shuffles the examples
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
    '''
    Splits the dataset intn n_folds
    :param X: The input matrix, N x M
    :param n_folds: The number of folds
    :param filename: The output file prefix
    :param write_to_csv: True if saving each fold required
    :return: An array of dictionaries of length n_folds
    '''
    N = len(X) // n_folds
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


def write_to_csv(to_df_object, filename, indexname=None):
    df_to_csv = pd.DataFrame(to_df_object)
    if indexname:
        df_to_csv.index.name = indexname
    df_to_csv.to_csv(filename + '.csv')


# Generate the files for 5-fold cross-validation
splits = cross_validation_split(data, 5, 'CandC', True)

MSEs_closed_form = []
all_weights = {}
fold = 1
for pair in splits:
    X_train = pair['train'].drop([M_cols - 1], axis=1)
    y_train = pair['train'][M_cols-1]
    X_test = pair['test'].drop([M_cols - 1], axis=1)
    y_test = pair['test'][M_cols - 1]

    weights = fit(X_train, y_train)
    MSEs_closed_form.append(mean_square_error(X_test, y_test, weights))
    all_weights['fold' + str(fold)] = weights
    fold += 1


MSEs_gd = []
all_weights_gd = {}
weights = np.random.uniform(high=10., size=[M_cols - 1])
fold = 1
for pair in splits:
    X_train = pair['train'].drop([M_cols - 1], axis=1)
    y_train = pair['train'][M_cols - 1]
    X_test = pair['test'].drop([M_cols - 1], axis=1)
    y_test = pair['test'][M_cols - 1]

    weights_gd = gradient_descent(np.array(X_train), np.array(y_train), epochs=20000, weights=np.copy(weights))
    MSEs_gd.append(mean_square_error(X_test, y_test, weights_gd))
    all_weights_gd['fold' + str(fold)] = weights_gd
    fold += 1


print('least squares:', np.average(MSEs_closed_form))
print('gradient descent:', np.average(MSEs_gd))

# Write weights to file (weight_index refers to w0, w1, etc. The weights for each fold are represented as columns in the csv. There are 5 columns and 123 rows)
write_to_csv(all_weights, 'q3_part2_weights_for_closed_form', 'weight_index')
write_to_csv(all_weights_gd, 'q3_part2_weights_for_gradient_descent', 'weight_index')


'''
Part 3) Ridge-regression: Using only least-squares closed form solution for this part.
'''
INCREMENTS = 100
MSE_vs_lambda = []
lambdas = np.arange(0, 5, 5 / INCREMENTS)
lowest_mse = 999999
best_lambda = -1

for lambda_value in lambdas:
    cur_mse = 0
    weights_for_folds = {}
    fold = 0
    for pair in splits:
        X_train = pair['train'].drop([M_cols - 1], axis=1)
        y_train = pair['train'][M_cols - 1]
        X_test = pair['test'].drop([M_cols - 1], axis=1)
        y_test = pair['test'][M_cols - 1]

        weights = fit(X_train, y_train, lambda_value)
        cur_mse += mean_square_error(X_test, y_test, weights)
        weights_for_folds['fold' + str(fold)] = weights

    cur_mse /= len(splits)
    if cur_mse < lowest_mse:
        lowest_mse = cur_mse
        best_lambda = lambda_value

    MSE_vs_lambda.append(cur_mse)
    write_to_csv(weights_for_folds, './question3part3/weights_lambda_' + str(lambda_value), 'weight_index')

print('Best lambda:', best_lambda)

for i in range(len(MSE_vs_lambda)):
    print('MSE for lambda =', lambdas[i], 'is', MSE_vs_lambda[i])

plt.figure(1)
plt.plot(lambdas[1:], MSE_vs_lambda[1:])
plt.title('Lambda vs MSE')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()

# Feature selection
# If a feature's associated weight is close to zero, it means that it contributes less to the prediction.
# These features are good candidates for being excluded from the model.
# We can first sort the weights and see which ones contribute the least to prediction
all_weights_feature_selection = {}
fold = 1
for pair in splits:
    X_train = pair['train'].drop([M_cols - 1], axis=1)
    y_train = pair['train'][M_cols - 1]
    X_test = pair['test'].drop([M_cols - 1], axis=1)
    y_test = pair['test'][M_cols - 1]

    weights = fit(X_train, y_train, lambda_reg=best_lambda) # Regularization with best_lambda from above
    all_weights_feature_selection['fold' + str(fold)] = weights
    fold += 1

# Now we look for weights that are close to zero in the regularized fit
features_to_drop = set()
for key in all_weights_feature_selection:
    w = all_weights_feature_selection[key]
    # This basically converts the weight vector [w0, w1...] to [(0, w0), (1, w1),...] so after we sort the weights we know which features they correspond to
    all_weights_feature_selection[key] = [(i, abs(w[i])) for i in range(len(w))]
    all_weights_feature_selection[key].sort(key=lambda x: x[1])
    [features_to_drop.add(x[0]) for x in all_weights_feature_selection[key][:20]]

for val in all_weights_feature_selection.values():
    print(val)

MSEs_final = []
all_weights_final = {}
fold = 1
for pair in splits:
    X_train = pair['train'].drop([M_cols - 1], axis=1)
    y_train = pair['train'][M_cols-1]
    X_test = pair['test'].drop([M_cols - 1], axis=1)
    y_test = pair['test'][M_cols - 1]

    # Drop the features we found to have near 0 coefficients
    X_train.drop(list(features_to_drop), axis=1, inplace=True)
    X_test.drop(list(features_to_drop), axis=1, inplace=True)

    weights = fit(X_train, y_train, lambda_reg=best_lambda)
    MSEs_final.append(mean_square_error(X_test, y_test, weights))
    all_weights_final['fold' + str(fold)] = weights
    fold += 1

print('Number of features to be dropped:', len(features_to_drop))
print('Best lambda:', best_lambda)
print('MSE for best lambda:', lowest_mse)
print('MSE for best lambda after dropping least contributing features:', np.average(MSEs_final))
