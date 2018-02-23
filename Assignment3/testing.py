import os, string, re, codecs, random
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

ds_path = './Datasets/'
types = ['train.txt', 'valid.txt', 'test.txt', ]


def confusion_matrix(pred, truth):
    # m = [[tn, fp], [fn, tp]]
    m = [[0, 0], [0, 0]]
    tp = np.dot(pred, truth)
    diff = pred - truth
    m[0][0] = np.count_nonzero(diff == 0) - tp
    m[0][1] = np.count_nonzero(diff == 1)
    m[1][0] = np.count_nonzero(diff == -1)
    m[1][1] = tp

    return m


def train_models(name, set, freq):
    n_folds = 5

    train = set['train']
    valid = set['valid']
    test = set['test']

    train_input = sparse.csr_matrix(train[0])
    valid_input = sparse.csr_matrix(valid[0])
    test_input = sparse.csr_matrix(test[0])

    train_truth = np.array(train[1])
    valid_truth = np.array(valid[1])
    test_truth = np.array(test[1])

    classes = len(np.unique(train_truth))
    average = 'micro'

    # Random Uniform Classifier
    pred = np.rint(np.random.random(len(train_truth)) * (classes - 1))
    print("{} Random Uniform Classifier train f1_score {}".format(name, f1_score(train_truth, pred, average=average)))

    pred = np.rint(np.random.random(len(valid_truth)) * (classes - 1))
    print("{} Random Uniform Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average=average)))

    pred = np.rint(np.random.random(len(test_truth)) * (classes - 1))
    print("{} Random Uniform Classifier test f1_score {}\n".format(name, f1_score(test_truth, pred, average=average)))

    # Majority Class Classifier
    maj = np.argmax(np.bincount(train_truth))

    pred = np.array([maj for i in range(len(train_truth))])
    print("{} Majority Class Classifier trian f1_score {}".format(name, f1_score(train_truth, pred, average=average)))

    pred = np.array([maj for i in range(len(valid_truth))])
    print("{} Majority Class Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average=average)))

    pred = np.array([maj for i in range(len(test_truth))])
    print("{} Majority Class Classifier test f1_score {}\n".format(name, f1_score(test_truth, pred, average=average)))

    # Naive Bayes
    alpha = np.arange(0.6, 0.8, 0.01)
    tuned_parameters = [{'alpha': alpha}]

    if freq:
        clf = MultinomialNB() if classes > 2 else BernoulliNB()
        clf = GridSearchCV(clf, tuned_parameters, cv=n_folds, refit=True)
        clf.fit(train_input, train_truth)

    else:
        clf = GaussianNB()

    clf.fit(train_input, train_truth)

    pred = clf.predict(train_input)
    print("{} Naive Bayes Classifier train f1_score {}".format(name, f1_score(train_truth, pred, average=average)))

    pred = clf.predict(valid_input)
    print("{} Naive Bayes Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average=average)))

    pred = clf.predict(test_input)
    print("{} Naive Bayes Classifier test f1_score {}".format(name, f1_score(test_truth, pred, average=average)))
    print(clf.best_params_, "\n")

    # Decision Tree
    tuned_parameters = [{'max_depth': [i for i in range(10, 20)], 'max_features': [1000 * i for i in range(2, 7)],
                         'max_leaf_nodes': [1000 * i for i in range(3, 6)]}]

    clf = DecisionTreeClassifier()
    clf = GridSearchCV(clf, tuned_parameters, cv=n_folds, refit=True)
    clf.fit(train_input, train_truth)

    pred = clf.predict(train_input)
    print("{} Decision Tree Classifier train f1_score {}".format(name, f1_score(train_truth, pred, average=average)))

    pred = clf.predict(valid_input)
    print("{} Decision Tree Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average=average)))

    pred = clf.predict(test_input)
    print("{} Decision Tree Classifier test f1_score {}".format(name, f1_score(test_truth, pred, average=average)))
    print(clf.best_params_, "\n")

    # Linear SVM
    tuned_parameters = [{'max_iter': [500 * i for i in range(5)], }]

    clf = LinearSVC()
    clf = GridSearchCV(clf, tuned_parameters, cv=n_folds, refit=True)
    clf.fit(train_input, train_truth)

    pred = clf.predict(train_input)
    print("{} Linear SVM Classifier train f1_score {}".format(name, f1_score(train_truth, pred, average=average)))

    pred = clf.predict(valid_input)
    print("{} Linear SVM Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average=average)))

    pred = clf.predict(test_input)
    print("{} Linear SVM Classifier test f1_score {} ".format(name, f1_score(test_truth, pred, average=average)))
    print(clf.best_params_, "\n")


def preprocess(file):
    translator = str.maketrans(" ", " ", string.punctuation)
    with open(file, 'r', encoding="utf-8") as f:
        text = f.read()
    text = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator)
    return text


''' DATA PARSING '''


def feature_extraction(set, n):
    file = preprocess(ds_path + set + types[0])
    word_list = file.split(" ")
    counter = Counter(word_list).most_common(n)
    dict = {}

    writer = open(set.split('-')[0] + '-vocab.txt', 'w')

    # save top words
    for i in range(n):
        word = counter[i][0]
        dict[word] = i + 1

        text = ("{}\t{}\t{}\n".format(word, i + 1, counter[i][1]))
        writer.write(text)

    for type in types:
        print(ds_path + set + type)
        file = preprocess(ds_path + set + type)

        examples = file.split("\n")[:-1]
        ds_output = [i[-1] for i in examples]

        writer = open(set.split('-')[0] + '-' + type.split('.')[0] + '.txt', 'w')
        for i in range(len(examples)):
            text = ""
            for word in examples[i].split(' ')[:-1]:
                if word in dict.keys():
                    text = "{} {}".format(text, dict[word])
            if len(text) == 0: text = ' '
            text = "{}\t{}\n".format(text, ds_output[i])
            writer.write(text[1:])

    return dict


def get_bow(dict, set):
    bow = {}
    bow_f = {}
    for type in types:
        name = type.split('.')[0]
        text = preprocess(ds_path + set + type).split('\n')

        text = list(filter(None, text))

        output = [int(line[-1]) for line in text]
        examples = [line[:-1] for line in text]

        vectorizer = CountVectorizer(vocabulary=dict.keys())

        vectors = vectorizer.fit_transform(examples)

        freq = normalize(vectors)
        vectors[vectors > 1] = 1
        binary = vectors

        bow[name] = [binary, output]
        bow_f[name] = [freq, output]

    return bow, bow_f


if __name__:
    n = 10000
    sets = ['yelp-', 'IMDB-']

    # ============== yelp ================
    set = sets[0]
    vocab_list = feature_extraction(set, n)
    yelp_bow, yelp_bowf = get_bow(vocab_list, set)

    print("\nUsing the BINARY Bag of Words")
    train_models(set, yelp_bow, False)

    print("\nUsing the Frequency Bag of Words")
    train_models(set, yelp_bowf, True)

    # ============== IMDB ================
    # set = sets[1]
    # vocab_list = feature_extraction(set, n)
# IMDB_bow, IMDB_bowf = get_bow(vocab_list, set)

# print("\nUsing the BINARY Bag of Words")
# train_models(set, IMDB_bow, False)

# print("\nUsing the Frequency Bag of Words")
# train_models(set, IMDB_bowf, True)