import numpy as np
import pandas as pd

'''
Question 3
'''


def write_to_csv(array_of_df_objects, filename, indexname=None, header=False, index=False):
    if not array_of_df_objects:
        return
    df_to_csv = array_of_df_objects if isinstance(array_of_df_objects[0], pd.DataFrame) else [pd.DataFrame(df) for df in array_of_df_objects]
    df_to_csv = pd.concat(df_to_csv, axis=0)
    if indexname:
        df_to_csv.index.name = indexname
    df_to_csv.to_csv(filename + '.csv', header=header, index=index)


N_examples = 2000
M = 20
test_set_percentage = 0.3
s1, s2, s3 = int(N_examples * .1), int(N_examples * .42), int(N_examples * .48)

# Read in data
ds2_cov1 = pd.read_csv(r'./Datasets/DS2_Cov1.txt', header=None).drop([M], axis=1)
ds2_cov2 = pd.read_csv(r'./Datasets/DS2_Cov2.txt', header=None).drop([M], axis=1)
ds2_cov3 = pd.read_csv(r'./Datasets/DS2_Cov3.txt', header=None).drop([M], axis=1)

ds2_c1_m1 = pd.read_csv(r'./Datasets/DS2_c1_m1.txt', header=None).drop([M], axis=1)
ds2_c1_m2 = pd.read_csv(r'./Datasets/DS2_c1_m2.txt', header=None).drop([M], axis=1)
ds2_c1_m3 = pd.read_csv(r'./Datasets/DS2_c1_m3.txt', header=None).drop([M], axis=1)

ds2_c2_m1 = pd.read_csv(r'./Datasets/DS2_c2_m1.txt', header=None).drop([M], axis=1)
ds2_c2_m2 = pd.read_csv(r'./Datasets/DS2_c2_m2.txt', header=None).drop([M], axis=1)
ds2_c2_m3 = pd.read_csv(r'./Datasets/DS2_c2_m3.txt', header=None).drop([M], axis=1)

data1_pos = np.random.multivariate_normal(np.squeeze(ds2_c1_m1), ds2_cov1, s1)
data1_neg = np.random.multivariate_normal(np.squeeze(ds2_c2_m1), ds2_cov1, s1)
data2_pos = np.random.multivariate_normal(np.squeeze(ds2_c1_m2), ds2_cov2, s2)
data2_neg = np.random.multivariate_normal(np.squeeze(ds2_c2_m2), ds2_cov2, s2)
data3_pos = np.random.multivariate_normal(np.squeeze(ds2_c1_m3), ds2_cov3, s3)
data3_neg = np.random.multivariate_normal(np.squeeze(ds2_c2_m3), ds2_cov3, s3)

positive_class = pd.DataFrame(np.concatenate((data1_pos, data2_pos, data3_pos)))
negative_class = pd.DataFrame(np.concatenate((data1_neg, data2_neg, data3_neg)))

positive_class[M] = 1
negative_class[M] = 0

positive_class = positive_class.sample(frac=1).reset_index(drop=True)
negative_class = negative_class.sample(frac=1).reset_index(drop=True)

test_pos = positive_class.iloc[:int(N_examples * test_set_percentage)]
test_neg = negative_class.iloc[:int(N_examples * test_set_percentage)]
train_pos = positive_class.iloc[int(N_examples * test_set_percentage):]
train_neg = negative_class.iloc[int(N_examples * test_set_percentage):]

write_to_csv([test_pos, test_neg], r'Datasets/DS2_test')
write_to_csv([train_pos, train_neg], r'Datasets/DS2_train')

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

def k_nearest_neighbors(train_x, train_y, test, k):
    if k & 1 == 0:
        k += 1
    train_x = train_x.values
    train_y = train_y.values
    test = test.values
    predictions = []
    for i in range(len(test)):
        knn_list = [(train_y[j], euclidean_dist(train_x[j], test[i])) for j in range(len(train_x))]
        top_k = sorted(knn_list, key=lambda x: x[1])[:k]
        count_pred = sum(x[0] == 1 for x in top_k)
        predictions.append(1 if count_pred > k - count_pred else 0)

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

'''
Question 5
'''
test_df = pd.read_csv(r'./Datasets/DS2_test.csv', header=None).astype(np.float64)
train_df = pd.read_csv(r'./Datasets/DS2_train.csv', header=None).astype(np.float64)
data = [train_df, test_df]


# LDA
mu_0, mu_1, cov_matrix, pi, w0, w = extract_lda_params(train_df)

prediction = predict(test_df.drop([20], axis=1), w0, w)
acc, precision, recall, f_measure, confusion = score(test_df[20], prediction)

print('LDA')
print('Confusion Matrix:', confusion)
print('Accuracy:', acc)
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', f_measure)
print()

# KNN
best_accuracy = -999999
best_precision = -999999
best_recall = -999999
best_f = -999999
ks = [0, 0, 0, 0]
for k in range(1, 69, 2):
    knn_pred = k_nearest_neighbors(train_df.drop([20], axis=1), train_df[20], test_df.drop([20], axis=1), k)
    acc, precision, recall, f_measure, confusion = score(test_df[20], knn_pred)

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

    print('KNN k:', k)
    print('Confusion Matrix:', confusion)
    print('Accuracy:', acc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F-Measure:', f_measure)
    print()

print(best_accuracy, best_precision, best_recall, best_f)
print(ks)
