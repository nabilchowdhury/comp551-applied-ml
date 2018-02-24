# Base libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy

# Processing & feature generation
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import normalize

# Classifiers
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

'''
Dataset Summary:
Yelp:
    Training set: 7000
    Valid set: 1000
    Test set: 2000

    Type: 5 class classification problem (1:worst - 5:best)

IMDB:
    Training set: 15000
    Valid set: 10000
    Test set: 25000

    Type: 2 class problem (1: positive, 0: negative)
'''

PATH = r'./Datasets'
M_FEATURES = 10000


def read_dataset(filename, column_names=['Review', 'Label']):
    return pd.read_table(os.path.join(PATH, filename), sep='\t', lineterminator='\n', header=None, names=column_names)


def preprocess(dataframe, column):
    dataframe[column] = dataframe[column].str.replace('<br /><br />', ' ').str.replace('[^\w\s]', '').str.lower()


def get_vocabulary(training_dict, column, features=M_FEATURES, save_to_file=False):
    most_common = {}
    for dataset in training_dict:
        all_words_list = [word for sentence in training_dict[dataset][column].str.split().tolist() for word in sentence]
        top_k = Counter(all_words_list).most_common(features)
        most_common[dataset] = {word[0]: i for i, word in enumerate(top_k)}

        if save_to_file:
            # Write to file for submission
            vocab = pd.DataFrame(top_k)
            vocab[2] = np.arange(0, features)  # These are the word IDs
            vocab.to_csv('./Submission/' + dataset + '-vocab.txt', sep='\t', header=False, index=False, columns=[0, 2, 1])

    return most_common


def bag_of_words(datasets, vocabulary, xname='Review', yname='Label'):
    binary_bog = {}
    freq_bog = {}
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    for name in datasets:
        vec = vectorizer.fit_transform(datasets[name][xname])
        freq_bog[name] = [normalize(vec), datasets[name][yname]]
        vec[vec > 1] = 1
        binary_bog[name] = [vec, datasets[name][yname]]

    return binary_bog, freq_bog


def write_converted_dataset(datasets, vocab_dict, dataset_name):
    for dataset in datasets[dataset_name]:
        with open('./Submission/' + dataset_name + '-' + dataset + '.txt', 'w') as file:
            for i in range(len(datasets[dataset_name][dataset])):
                file.write(' '.join([str(vocab_dict[dataset_name][word]) for word in datasets[dataset_name][dataset].iloc[i, 0].split()
                                     if word in vocab_dict[dataset_name]]) + '\t' + str(datasets[dataset_name][dataset].iloc[i, 1]) + '\n')


def random_classifier(train_on, predict_on):
    return np.random.choice(np.unique(train_on[1]), len(predict_on[1]))


def majority_class_classifier(train_on, predict_on):
    return np.full(len(predict_on[1]), scipy.stats.mode(train_on[1])[0][0])


def print_format(s, *args):
    print(s.format(*args))


def do_clf_test(clf, dataset_dict, tune_params=None, average='micro'):
    name = clf.__name__.upper() if callable(clf) else clf.__class__.__name__.upper()
    if not callable(clf):
        if tune_params is not None:
            # Set up GridSearch with validation set
            total_length = len(dataset_dict['train'][1]) + len(dataset_dict['valid'][1])
            ps = PredefinedSplit(test_fold=[-1 if i < len(dataset_dict['train'][1]) else 0 for i in range(total_length)])
            clf = GridSearchCV(clf, tune_params, cv=ps, refit=True)
            train_x = scipy.sparse.vstack([dataset_dict['train'][0], dataset_dict['valid'][0]])
            train_y = np.concatenate([dataset_dict['train'][1], dataset_dict['valid'][1]])
        else:
            train_x = dataset_dict['train'][0]
            train_y = dataset_dict['train'][1]

        clf.fit(train_x, train_y)

    for dname, dset in dataset_dict.items():
        score = f1_score(dset[1], clf(dataset_dict['train'], dset) if callable(clf) else clf.predict(dset[0]), average=average)
        print_format('\t{} score using {}: {}\n', dname.upper(), name, score)


if __name__ == '__main__':
    WRITE = False
    # Yelp datsets
    yelp_train = read_dataset('yelp-train.txt')
    yelp_valid = read_dataset('yelp-valid.txt')
    yelp_test = read_dataset('yelp-test.txt')

    # IMDB datasets
    imdb_train = read_dataset('IMDB-train.txt')
    imdb_valid = read_dataset('IMDB-valid.txt')
    imdb_test = read_dataset('IMDB-test.txt')

    # yelp
    datasets = {
        'yelp': {'train': yelp_train, 'valid': yelp_valid, 'test': yelp_test},
        'imdb': {'train': imdb_train, 'valid': imdb_valid, 'test': imdb_test}
    }

    # Group sets
    training = {'yelp': yelp_train, 'imdb': imdb_train}
    valid = {'yelp': yelp_valid, 'imdb': imdb_valid}
    test = {'yelp': yelp_test, 'imdb': imdb_test}

    # Sanity Check
    print('CHECK DATA:')
    for name, training_set in training.items():
        print(name.upper(), 'size:', str(len(training_set)))
        print(training_set.head(), '\n', '-' * 80)

    '''
    Question 1: Preprocessing
    '''
    for s in datasets.values():
        for df in s.values():
            preprocess(df, 'Review')

    # Verify preprocessing
    print('\nAFTER PREPROCESSING:')
    for name, training_set in training.items():
        print(training_set.head(), '\n', '-' * 80)

    # Generate vocabulary for yelp and imdb datasets from training data, and write to file
    vocabulary = get_vocabulary(training, 'Review', M_FEATURES, WRITE)

    # Bag of words
    yelp_binary, yelp_freq = bag_of_words(datasets['yelp'], vocabulary['yelp'])
    imdb_binary, imdb_freq = bag_of_words(datasets['imdb'], vocabulary['imdb'])

    # Write converted datasets to file
    if WRITE:
        write_converted_dataset(datasets, vocabulary, 'yelp')
        write_converted_dataset(datasets, vocabulary, 'imdb')


    # run_predictions([random_classifier, majority_class_classifier, LinearSVC(), DecisionTreeClassifier()])

    '''
    Question 2: Yelp binary bag of words with hyperparameter turning using GridSearchCV :)
    '''
    print('Yelp Binary Bag of Words Performances')
    # do_clf_test(random_classifier, yelp_binary)
    # do_clf_test(majority_class_classifier, yelp_binary)
    do_clf_test(BernoulliNB(), yelp_binary, [{'alpha': [pow(10, -i) for i in range(6)]}])
    # do_clf_test(DecisionTreeClassifier(), yelp_binary)
    # do_clf_test(LinearSVC(), yelp_binary)

    '''
    Question 3: Yelp frequency bag of words
    '''
    # print('Yelp Frequency Bag of Words Performances')
    # do_clf_test(random_classifier, yelp_freq)
    # do_clf_test(majority_class_classifier, yelp_freq)
    # # do_clf_test(GaussianNB(), yelp_freq)
    # do_clf_test(DecisionTreeClassifier(), yelp_freq)
    # do_clf_test(LinearSVC(), yelp_freq)
