import numpy as np
import pandas as pd

'''
Question 1
'''


def write_to_csv(array_of_df_objects, filename, indexname=None, header=False, index=False):
    if not array_of_df_objects:
        return
    df_to_csv = array_of_df_objects if isinstance(array_of_df_objects[0], pd.DataFrame) else [pd.DataFrame(df) for df in array_of_df_objects]
    df_to_csv = pd.concat(df_to_csv, axis=0)
    if indexname:
        df_to_csv.index.name = indexname
    df_to_csv.to_csv(filename + '.csv', header=header, index=index)


# Read in data
N_examples = 2000
M = 20
test_set_percentage = 0.3

covariance_matrix = pd.read_csv(r'./Datasets/DS1_Cov.txt', header=None).drop([M], axis=1)
mean_1 = pd.read_csv(r'./Datasets/DS1_m_1.txt', header=None).drop([M], axis=1)
mean_0 = pd.read_csv(r'./Datasets/DS1_m_0.txt', header=None).drop([M], axis=1)

positive_class = pd.DataFrame(np.random.multivariate_normal(np.squeeze(mean_1), covariance_matrix, N_examples))
negative_class = pd.DataFrame(np.random.multivariate_normal(np.squeeze(mean_0), covariance_matrix, N_examples))

positive_class[M] = 1
negative_class[M] = 0

positive_class = positive_class.sample(frac=1).reset_index(drop=True)
negative_class = negative_class.sample(frac=1).reset_index(drop=True)

test_pos = positive_class.iloc[:int(N_examples * test_set_percentage)]
test_neg = negative_class.iloc[:int(N_examples * test_set_percentage)]
train_pos = positive_class.iloc[int(N_examples * test_set_percentage):]
train_neg = negative_class.iloc[int(N_examples * test_set_percentage):]

write_to_csv([test_pos, test_neg], r'Datasets/DS1_test')
write_to_csv([train_pos, train_neg], r'Datasets/DS1_train')
