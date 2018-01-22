import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
1) MODEL SELECTION
'''
'''
Setup dataset
'''
GRANULARITY = 500 # Used for plot smoothness (higher is better)
FIGSIZE = [9, 7]

# Load the examples
train_1 = pd.read_csv(r'./Datasets/Dataset_1_train.csv', header=None)
valid_1 = pd.read_csv(r'./Datasets/Dataset_1_valid.csv', header=None)
test_1 = pd.read_csv(r'./Datasets/Dataset_1_test.csv', header=None)

# This makes it easier to work on all sets simultaneously
example_set = {'train': train_1, 'valid': valid_1, 'test': test_1}
output_set = {}

'''
Part 1: Fit 20 degree polynomial
'''
MAX_DEGREE = 20

for key in example_set:
    # example_set[key].sort_values(by=[0], inplace=True)
    # Separate y from features
    output_set[key] = example_set[key][1]
    example_set[key].drop([1, 2], axis=1, inplace=True) # drop y column and null column from original dataframe
    # Add column of 1s as bias
    example_set[key].columns = [1]
    example_set[key][0] = 1.
    example_set[key] = example_set[key][[0, 1]]

# Generate polynomial features up to x^20. We already have bias and x^1
for i in range(2, MAX_DEGREE + 1):
    for _, df in example_set.items():
        df[i] = np.power(df[1], i)

for key in example_set:
    example_set[key] = np.array(example_set[key], dtype=np.float64)
    output_set[key] = np.array(output_set[key], dtype=np.float64)

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


def visualize(W, title=""):
    x_axis = pd.DataFrame(np.ones(GRANULARITY))
    x_axis[1] = np.arange(-1, 1, 2 / GRANULARITY)
    for i in range(2, MAX_DEGREE + 1):
        x_axis[i] = pow(x_axis[1], i)
    decision_boundary = np.matmul(x_axis, W)

    plt.figure(1, figsize=FIGSIZE)
    for i, key in enumerate(example_set, 1):
        plt.subplot(3, 1, i)
        plt.scatter(example_set[key][:, 1], output_set[key])
        plt.plot(x_axis[1], decision_boundary, 'r--')
        plt.title(title + key.upper())
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim([-50, 50])
    plt.tight_layout()

    # plt.figure(2)
    # plt.scatter(example_set['train'][:, 1], output_set['train'])
    # plt.plot(x_axis[1], decision_boundary, 'r--')
    # plt.title('Unregularized fit on Training set')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.ylim([-50, 50])

    plt.show()


# Calculate the optimal weights based on the TRAINING examples
weights = fit(example_set['train'], output_set['train'])

# Display MSEs for each set
for key in example_set:
    print('MSE for', key, mean_square_error(example_set[key], output_set[key], weights))

# Observe the curve
visualize(weights, 'Part 1: ')

'''
Part 2: Add L2 regularization
'''
INCREMENTS = 10000

train_mse = []
valid_mse = []
best_lambda = 9999999
lowest_mse = 9999999
for i in range(INCREMENTS + 1):
    # Calculate w with on the TRAINING set using regularization
    w = fit(example_set['train'], output_set['train'], 1 / INCREMENTS * i)
    # Use w to compute MSE
    mse_train = mean_square_error(example_set['train'], output_set['train'], w)
    mse_valid = mean_square_error(example_set['valid'], output_set['valid'], w)

    train_mse.append(mse_train)
    valid_mse.append(mse_valid)

    # Track the best lambda
    if mse_valid < lowest_mse:
        lowest_mse = mse_valid
        best_lambda = 1/INCREMENTS * i

print('Best Lambda:', best_lambda)

# Calculate optimal weights using best_lambda
OPT_WEIGHTS = fit(example_set['train'], output_set['train'], best_lambda)
# Test performance
test_mse = mean_square_error(example_set['test'], output_set['test'] , OPT_WEIGHTS)
print('MSE of test set:', test_mse)

# Plot Lambda vs MSE for train and valid sets
plt.figure(figsize=FIGSIZE)
plt.plot(np.arange(0, 1, 1/INCREMENTS), train_mse[1:], label='train')
plt.plot(np.arange(0, 1, 1/INCREMENTS), valid_mse[1:], label='valid')
plt.plot([best_lambda], [lowest_mse], marker='o', markersize=6, color='r')
plt.title("Part 2: Lambda vs MSE")
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()
plt.savefig('q1p14.png')
plt.show()

# Plot chosen model
visualize(OPT_WEIGHTS, 'Part 2: ')
