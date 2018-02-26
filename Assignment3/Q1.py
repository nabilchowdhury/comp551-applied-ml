# Base libraries
import numpy as np
import os
import pandas as pd
import scipy

# Processing & feature generation
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import normalize

# Classifiers
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
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

# Constants
PATH = r'./Datasets'
M_FEATURES = 10000


def read_dataset(filename, column_names=['Review', 'Label']):
    '''
    Parses the file into a pandas dataframe object.
    :param filename: The file to be parsed
    :param column_names: The desired column names of the dataframe
    :return: The file as a dataframe object
    '''
    return pd.read_table(os.path.join(PATH, filename), sep='\t', lineterminator='\n', header=None, names=column_names)


def preprocess(dataframe, column):
    '''
    Preprocesses the reviews by stripping away all non-word, non-space characters. Additionally removes <br /> tags for IMDB set
    :param dataframe: The dataframe object
    :param column: The column to be preprocessed. This will be 'Review' for this assignment.
    :return: None
    '''
    dataframe[column] = dataframe[column].str.replace('<br /><br />', ' ').str.replace('[^\w\s]', '').str.lower()


def get_vocabulary(training_dict, column, features=M_FEATURES, save_to_file=False):
    '''
    For each training set in training_dict, return the corresponding vocabulary to be used as the feature set.
    The method first counts the frequencies of all words across each training set, and then chooses the top most
    frequent words as the feature set.
    :param training_dict: The dictionary of training sets. We have yelp and IMDB training sets.
    :param column: The column to get the words from. This is 'Review'
    :param features: The number of top most frequent features to be used
    :param save_to_file: If True, saves the feature set, along with the frequencies and IDs to their corresponding file
                         as required by the assignment
    :return: Dictionary of vocabularies for both yelp and IMDB
    '''
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
    '''
    Converts each dataset in datasets to both binary and frequency bag of words representations.
    :param datasets: Dictionary of datasets to be converted
    :param vocabulary: The vocabulary extracted by :func:`get_vocabulary`
    :param xname: The name of the feature column ('Review')
    :param yname: The name of the label column ('Label')
    :return: Binary and frequency BoW dictionaries. Each dictionary has keys corresponding to the keys of datasets.
    '''
    binary_bow = {}
    freq_bow = {}
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    for name in datasets:
        vec = vectorizer.fit_transform(datasets[name][xname])
        freq_bow[name] = [normalize(vec), datasets[name][yname]]
        vec[vec > 1] = 1
        binary_bow[name] = [vec, datasets[name][yname]]

    return binary_bow, freq_bow


def write_converted_dataset(datasets, vocab_dict, dataset_name):
    '''
    Replaces the words of the datasets by their unique IDs from the vocabulary, as required by the assignment. It then
    writes the converted datasets to file.
    :param datasets: Dictionary containing the datasets to be converted
    :param vocab_dict: The vocabulary dictionary obtained from :func:`get_vocabulary`
    :param dataset_name: 'yelp' or 'imdb'
    :return: None
    '''
    for dataset in datasets[dataset_name]:
        with open('./Submission/' + dataset_name + '-' + dataset + '.txt', 'w') as file:
            for i in range(len(datasets[dataset_name][dataset])):
                file.write(' '.join([str(vocab_dict[dataset_name][word]) for word in datasets[dataset_name][dataset].iloc[i, 0].split()
                                     if word in vocab_dict[dataset_name]]) + '\t' + str(datasets[dataset_name][dataset].iloc[i, 1]) + '\n')


def random_classifier(train_on, predict_on):
    '''
    Classifies predict_on into a random class.
    :param train_on: The training set
    :param predict_on: The test set
    :return: Predicted labels of the test set
    '''
    return np.random.choice(np.unique(train_on[1]), len(predict_on[1]))


def majority_class_classifier(train_on, predict_on):
    '''
    Classifies predict_on into the majority class (mode) of the training set.
    :param train_on: The training set
    :param predict_on: The test set
    :return: Predicted labels of the test set
    '''
    return np.full(len(predict_on[1]), scipy.stats.mode(train_on[1])[0][0])


def print_format(s, *args):
    '''
    Helper to print formatted strings
    :param s: The unformatted string with placeholders
    :param args: The args that go into the placeholders of s
    :return: None
    '''
    print(s.format(*args))


def do_clf_test(clf, dataset_dict, tune_params=None, tune=True, average='micro'):
    '''
    Fits the classifier clf to the training set, and predicts on the test set. Prints the F1 scores for each set.
    :param clf: An sklearn classifier or :func:`random_classifier` or :func:`majority_class_classifier`
    :param dataset_dict: Dictionary of datasets to predict on (must contain keys 'train', 'valid', 'test')
    :param tune_params: If not None, will tune the parameters on the validation set before predicting on test. Tuning
                        is done using sklearn's GridSearchCV
    :param tune: Must be true in addition to tune_params not being None to perform tuning
    :param average: The average paramters of sklearn's f1_score. Defaults to 'micro'
    :return: None
    '''
    name = clf.__name__.upper() if callable(clf) else clf.__class__.__name__.upper()
    print_format('\tScores using {}:', name)

    if not callable(clf):
        train_x = dataset_dict['train'][0]
        train_y = dataset_dict['train'][1]
        if tune and tune_params is not None:
            # Set up GridSearch with validation set
            valid_x, valid_y = dataset_dict['valid'][0], dataset_dict['valid'][1]
            ps = PredefinedSplit(test_fold=[-1 if i < len(train_y) else 0 for i in range(len(train_y) + len(valid_y))])
            clf = GridSearchCV(clf, tune_params, cv=ps, n_jobs=2)
            train_x = scipy.sparse.vstack([train_x, valid_x])
            train_y = np.concatenate([train_y, valid_y])

        clf.fit(train_x, train_y)
        if tune and tune_params is not None: print('\t\tBest params:', clf.best_params_)

    for dname, dset in dataset_dict.items():
        score = f1_score(dset[1], clf(dataset_dict['train'], dset) if callable(clf) else clf.predict(dset[0]), average=average)
        print_format('\t\t{}: {}', dname.upper(), score)


if __name__ == '__main__':
    # Useful flags
    WRITE = False  # If true, write to file
    PERFORM_TUNING = True

    # Yelp datsets
    yelp_train = read_dataset('yelp-train.txt')
    yelp_valid = read_dataset('yelp-valid.txt')
    yelp_test = read_dataset('yelp-test.txt')

    # IMDB datasets
    imdb_train = read_dataset('IMDB-train.txt')
    imdb_valid = read_dataset('IMDB-valid.txt')
    imdb_test = read_dataset('IMDB-test.txt')

    # All sets
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


    '''
    Question 2: Question 2: Yelp Binary BoW f1 scores on random uniform classifier, majority-class classifier, 
    Naive Bayes, Decision Trees, LinearSVM w/ hyperparameter tuning using GridSearchCV
    '''
    print('Yelp Binary Bag of Words Performances')
    do_clf_test(random_classifier, yelp_binary)  # Random uniform classifier
    do_clf_test(majority_class_classifier, yelp_binary)  # Majority class classifier

    # BernoulliNB
    params = {'alpha': np.arange(0.01, 1.01, 0.01)}
    do_clf_test(BernoulliNB(), yelp_binary, params, tune=PERFORM_TUNING)

    # Decision Tree
    params = {'max_depth': np.arange(13, 17),
              'max_features': np.arange(0.1, 0.5, 0.1),
              'min_samples_leaf': np.arange(3, 6)}
    do_clf_test(DecisionTreeClassifier(), yelp_binary, params, tune=PERFORM_TUNING)

    # Linear SVM
    params = {'C': np.logspace(-2, 2, num=8),
              'max_iter': np.arange(10, 100, 10)}
    do_clf_test(LinearSVC(), yelp_binary, params, tune=PERFORM_TUNING)

    '''
    Question 3: Yelp Frequency BoW
    '''
    print('Yelp Frequency Bag of Words Performances')
    do_clf_test(random_classifier, yelp_freq)
    do_clf_test(majority_class_classifier, yelp_freq)

    # GaussianNB: Requires dense arrays
    do_clf_test(GaussianNB(), {key: [value[0].toarray(), value[1]] for key, value in yelp_freq.items()})

    # Decision Tree
    params = {'max_depth': np.arange(13, 17),
              'max_features': np.arange(0.1, 0.5, 0.1),
              'min_samples_leaf': np.arange(3, 6)}
    do_clf_test(DecisionTreeClassifier(), yelp_freq, params, tune=PERFORM_TUNING)

    # Linear SVM
    params = {'C': np.logspace(-2, 2, num=8),
              'max_iter': np.arange(10, 100, 10)}
    do_clf_test(LinearSVC(), yelp_freq, params, tune=PERFORM_TUNING)

    '''
    Question 4: Repeat Q2 and Q3 with IMDB
    '''
    print('IMDB Binary Bag of Words Performances')
    do_clf_test(random_classifier, imdb_binary)
    # Majority class classifier doesn't make sense for IMDB since it is a balanced dataset

    # BernoulliNB
    params = {'alpha': np.arange(0.01, 1.01, 0.01)}
    do_clf_test(BernoulliNB(), imdb_binary, params, tune=PERFORM_TUNING)

    # Decision Tree
    params = {'max_depth': np.arange(13, 17),
              'max_features': np.arange(0.1, 0.5, 0.1),
              'min_samples_leaf': np.arange(3, 6)}
    do_clf_test(DecisionTreeClassifier(), imdb_binary, params, tune=PERFORM_TUNING)

    # Linear SVM
    params = {'C': np.logspace(-2, 2, num=8),
              'max_iter': np.arange(10, 100, 10)}
    do_clf_test(LinearSVC(), imdb_binary, params, tune=PERFORM_TUNING)

    print('IMDB Frequency Bag of Words Performances')
    do_clf_test(random_classifier, imdb_freq)

    # GaussianNB: Requires dense arrays
    # No parameters to tune
    do_clf_test(GaussianNB(), {key: [value[0].toarray(), value[1]] for key, value in imdb_freq.items()})

    # Decision Tree
    params = {'max_depth': np.arange(13, 17),
              'max_features': np.arange(0.1, 0.5, 0.1),
              'min_samples_leaf': np.arange(3, 6)}
    do_clf_test(DecisionTreeClassifier(), imdb_freq, params, tune=PERFORM_TUNING)

    # Linear SVM
    params = {'C': np.logspace(-2, 2, num=8),
              'max_iter': np.arange(10, 100, 10)}
    do_clf_test(LinearSVC(), imdb_freq, params, tune=PERFORM_TUNING)