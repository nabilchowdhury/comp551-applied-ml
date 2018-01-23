import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

'''
2) Gradient Descent for Regression
'''
GRANULARITY = 50
MAX_DEGREE = 1

train_2 = pd.read_csv(r'./Datasets/Dataset_2_train.csv', header=None)
valid_2 = pd.read_csv(r'./Datasets/Dataset_2_valid.csv', header=None)
test_2 = pd.read_csv(r'./Datasets/Dataset_2_test.csv', header=None)

example_set = {'train': train_2, 'valid': valid_2, 'test': test_2}
output_set = {}


def mean_square_error(X, y, W):
    y_hat = np.matmul(X, W)
    mean_square_err = np.sum(np.square(y - y_hat)) / len(y)

    return mean_square_err

# Normal equation with regularization (lambda_reg=0 means no regularization)
def fit(X, y, lambda_reg=0):
    identity = np.identity(MAX_DEGREE + 1)
    # identity[0, 0] = 0 # We do not penalize the bias term

    X_square = np.matmul(np.transpose(X), X) + lambda_reg * identity
    X_square_inverse = np.linalg.pinv(X_square)
    weights = np.matmul(np.matmul(X_square_inverse, np.transpose(X)), y)

    return weights

for key in example_set:
    output_set[key] = example_set[key][1]
    example_set[key].drop([1, 2], axis=1, inplace=True)
    example_set[key].columns = [1]
    example_set[key][0] = 1.
    example_set[key] = example_set[key][[0, 1]]

for i in range(2, MAX_DEGREE + 1):
    for _, df in example_set.items():
        df[i] = np.power(df[1], i)

for key in example_set:
    example_set[key] = np.array(example_set[key], dtype=np.float64)
    output_set[key] = np.array(output_set[key], dtype=np.float64)

# Sanity check: Check against polyfit and our least squares fit to see if SGD is accurate
weights_opt = np.polyfit(example_set['train'][:, 1], output_set['train'], MAX_DEGREE)[::-1]
weights_normal = fit(example_set['train'], output_set['train'])
print('polyfit weights:', weights_opt)
print('normaleq weights:', weights_normal)

'''
Try gradient descent first for fun
'''
ALPHA = 1e-5
EPOCHS = 20000

weights_original = np.random.uniform(high=10., size=[MAX_DEGREE + 1])
# weights = np.array([4., 7.], dtype=np.float64)
print('weights original', weights_original)
X = example_set['train']
y = output_set['train']
#
# for epoch in range(EPOCHS):
#     weights = weights - ALPHA * np.matmul( np.transpose(X), np.matmul(X, weights) - y)


'''
Stochastic gradient descent
'''

# Decision boundary x axis
x_axis = pd.DataFrame(np.ones(GRANULARITY))
x_axis[1] = np.arange(0, 2, 2 / GRANULARITY)
for i in range(2, MAX_DEGREE + 1):
    x_axis[i] = pow(x_axis[1], i)

# decision_boundary = np.matmul(x_axis, weights_opt)
# plt.scatter(example_set['train'][1], output_set['train'])
# plt.plot(x_axis[1], decision_boundary, 'y--')
# plt.show()


def SGD(X, y, epochs, weights, alpha, plot_at_epoch=None):
    N = example_set['train'].shape[0]
    train_mse = []
    valid_mse = []
    for epoch in range(epochs):
        if epoch % 2000 == 0:
            decision_boundary = np.matmul(x_axis, weights)
            plt.scatter(X[:, 1], y, s=10)
            plt.plot(x_axis[1], decision_boundary, 'r--')
            plt.title('Part 3: Fit after ' + str(epoch) + ' epochs')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim(ymin=0, ymax=12)
            plt.xlim(xmax=1.6)
            plt.show()
        for i in range(N):
            weights = weights - alpha * (np.sum((X[i] * weights)) - y[i]) * X[i]
        train_mse.append(mean_square_error(X, y, weights))
        valid_mse.append(mean_square_error(example_set['valid'], output_set['valid'], weights))

    return weights, train_mse, valid_mse


weights, train_mse, valid_mse = SGD(X, y, EPOCHS, weights_original.copy(), ALPHA)

# plt.figure(1)
# plt.plot(np.arange(0, EPOCHS, 1), train_mse, label='train')
# plt.plot(np.arange(0, EPOCHS, 1), valid_mse, label='valid')
# plt.title('Part 1: Learning curve for Train and Valid')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.ylim(ymin=0, ymax=2.5)
#
# '''
# Part 2) Choose best step size ALPHA using validation data
# '''
# best_alpha = -1
# lowest_mse = 99999
# alphas = [1e-6, 2e-6, 1e-5, 2e-5, 1e-4, 2e-4, 1e-3, 2e-3, 1e-2, 2e-2]
# MSEs = []
# # weights_og = np.random.uniform(high=10., size=[MAX_DEGREE + 1])
# weights_og = np.array([6.85035978, 8.34571567], dtype=np.float64)
# # print('weghts og', weights_og)
# for cur_alpha in alphas:
#     print('cur alpha:', cur_alpha)
#     weights, _, _ = SGD(X, y, EPOCHS, weights_og.copy(), cur_alpha)
#     mse_for_valid = mean_square_error(example_set['valid'], output_set['valid'], weights)
#     print(mse_for_valid)
#     print(weights)
#     MSEs.append(mse_for_valid)
#     if mse_for_valid < lowest_mse:
#         lowest_mse = mse_for_valid
#         best_alpha = cur_alpha
#
# plt.figure(2)
# plt.plot(alphas, MSEs, 'ro')
# plt.plot(alphas, MSEs, 'r--')
# plt.title('Part 2: Step size (alpha) vs MSE for Valid')
# plt.xlabel('Step size (alpha)')
# plt.ylabel('MSE')
# plt.ylim(ymax=0.09)
# plt.xlim([-0.00001, 0.0004])
# print(best_alpha)
#
# # Decision boundary x axis
# x_axis = pd.DataFrame(np.ones(GRANULARITY))
# x_axis[1] = np.arange(0, 2, 2 / GRANULARITY)
# for i in range(2, MAX_DEGREE + 1):
#     x_axis[i] = pow(x_axis[1], i)
#
# decision_boundary = np.matmul(x_axis, weights_opt)
# plt.scatter(example_set['train'][1], output_set['train'])
# plt.plot(x_axis[1], decision_boundary, 'y--')
# plt.show()

plt.show()
