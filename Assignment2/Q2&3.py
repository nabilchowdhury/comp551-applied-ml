import numpy as np
import pandas as pd

'''
Question 2
'''
test_df = pd.read_csv(r'./Datasets/DS1_test.csv', header=None).astype(np.float64)
train_df = pd.read_csv(r'./Datasets/DS1_train.csv', header=None).astype(np.float64)
data = [train_df, test_df]


def extract_lda_params(X):
    class_freq = X.groupby(20).size()
    # N0 is number of negative class data pts, N1 is number of positive class data pts (50/50 in this case)
    N0, N1 = class_freq.iloc[0], class_freq.iloc[1]
    pi = N1 / (N0 + N1)
    mu_1 = X.loc[X[20] == 1].mean().drop(20)
    mu_0 = X.loc[X[20] == 0].mean().drop(20)

    # Calculate covariance matrix
    normalized_pos = X.loc[X[20] == 1].drop([20], axis=1) - mu_1
    normalized_neg = X.loc[X[20] == 0].drop([20], axis=1) - mu_0

    S1 = 1 / N1 * normalized_pos.T.dot(normalized_pos)
    S0 = 1 / N0 * normalized_neg.T.dot(normalized_neg)
    cov_matrix = 1 / (N1 + N0) * (N1 * S1 + N0 * S0)

    inv_cov = np.linalg.inv(cov_matrix.values)
    w = np.matmul(inv_cov, mu_1 - mu_0)
    w0 = -1 / 2 * (np.matmul(np.matmul(mu_1.values, inv_cov), mu_1.values) -
                   np.matmul(np.matmul(mu_0.values, inv_cov), mu_0.values)) + \
                   np.log(pi / (1 - pi))

    return mu_0, mu_1, cov_matrix, pi, w0, w


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def predict(X, w0, w):
    hypothesis_raw = sigmoid(np.matmul(X, w) + w0)
    hypothesis = [1 if val > 0.5 else 0 for val in hypothesis_raw]
    return hypothesis


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def k_nearest_neighbors(train_x, train_y, test, k, testmode=False):
    if k & 1 == 0:
        k += 1
    train_x = train_x.values
    train_y = train_y.values
    test = test.values
    predictions = {}

    for i in range(len(test)):
        knn_list = [(train_y[j], euclidean_dist(train_x[j], test[i])) for j in range(len(train_x))]
        top_k = [pair[0] for pair in sorted(knn_list, key=lambda x: x[1])[:k]]
        top_k_cumsum = np.cumsum(top_k)
        # Basically, if we find k neighbors, we can calculate predictions for k from 1 to k - 1. This avoids
        # repeated computations and speeds up KNN for performance evaluation
        for neighbors in range(1 if testmode else k, k + 2, 2):
            if neighbors not in predictions:
                predictions[neighbors] = []
            predictions[neighbors].append(1 if top_k_cumsum[neighbors - 1] > neighbors - top_k_cumsum[neighbors - 1] else 0)

    return predictions


def score(y, h_y):
    EPSILON = 1e-5

    # Accuracy
    accuracy = 0
    for i in range(len(h_y)):
        accuracy += 1 if abs(h_y[i] - y[i]) < EPSILON else 0
    accuracy /= len(h_y)

    # Confusion Matrix
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(y)):
        if abs(h_y[i] - y[i]) < EPSILON:
            tp += 1 if y[i] == 1 else 0
            tn += 1 if y[i] == 0 else 0
        else:
            fp += 1 if y[i] == 1 else 0
            fn += 1 if y[i] == 0 else 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * precision * recall / (precision + recall)

    return accuracy,  precision, recall, f_measure, np.array([[tp, fp], [fn, tn]])


mu_0, mu_1, cov_matrix, pi, w0, w = extract_lda_params(train_df)

prediction = predict(test_df.drop([20], axis=1), w0, w)
acc, precision, recall, f_measure, confusion = score(test_df[20], prediction)

print('LDA')
print('Confusion Matrix:', confusion)
print('Accuracy:', acc)
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', f_measure)

# KNN
max_k = 99
accuracies = []
precisions = []
recalls = []
f_measures = []
best_accuracy = -999999
best_precision = -999999
best_recall = -999999
best_f = -999999
ks = [0, 0, 0, 0]
knn_pred = k_nearest_neighbors(train_df.drop([20], axis=1), train_df[20], test_df.drop([20], axis=1), max_k, True)
for k in range(1, max_k + 2, 2):
    acc, precision, recall, f_measure, confusion = score(test_df[20], knn_pred[k])
    accuracies.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    f_measures.append(f_measure)

    if acc > best_accuracy:
        best_accuracy = acc
        ks[0] = k
    if precision > best_precision:
        best_precision = precision
        ks[1] = k
    if recall > best_recall:
        best_recall = recall
        ks[2] = k
    if f_measure > best_f:
        best_f = f_measure
        ks[3] = k

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(np.arange(1, max_k + 2, 2), accuracies, label='accuracy')
plt.plot(np.arange(1, max_k + 2, 2), precisions, label='precision')
plt.plot(np.arange(1, max_k + 2, 2), recalls, label='recall')
plt.plot(np.arange(1, max_k + 2, 2), f_measures, label='f-measure')
plt.xlabel('K')
plt.ylabel('Metric Score')
plt.title('Metrics for Different values of k')
plt.legend()
plt.show()

all_res = [(k, accuracies[i], precisions[i], recalls[i], f_measures[i]) for i, k in enumerate(range(1, max_k + 2, 2))]

# Print for latex
print('k value', '&', 'Accuracy', '&', 'Precision', '&', 'Recall', '&', 'F-Measure' + '\\\\')
for tup in all_res:
    print(tup[0], '&', '% 0.5f' % tup[1], '&', '% 0.5f' % tup[2], '&', '% 0.5f' % tup[3], '&', '% 0.5f' % tup[4], '\\\\')

print(best_accuracy, best_precision, best_recall, best_f)
print(ks)
