import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def binary_bag_of_words(X, y):
    pass

def frequency_bag_of_words(X, y):
    pass

def preprocess_text(text, column):
    text[column] = text[column].str.replace('[^\w\s]', '').str.lower()

def frequency_count(text, sort_ascending=True):
    pass

if __name__ == '__main__':
    '''
    Read in the data
    '''
    # Yelp datsets
    yelp_train = pd.read_table(r'./Datasets/yelp-train.txt', sep='\t', lineterminator='\n', header=None,
                               names=['Review', 'Rating'])
    yelp_valid = pd.read_table(r'./Datasets/yelp-valid.txt', sep='\t', lineterminator='\n', header=None,
                               names=['Review', 'Rating'])
    yelp_test = pd.read_table(r'./Datasets/yelp-test.txt', sep='\t', lineterminator='\n', header=None,
                              names=['Review', 'Rating'])

    # IMDB datasets
    imdb_train = pd.read_table(r'./Datasets/IMDB-train.txt', sep='\t', lineterminator='\n', header=None,
                               names=['Review', 'Sentiment'])
    imdb_valid = pd.read_table(r'./Datasets/IMDB-valid.txt', sep='\t', lineterminator='\n', header=None,
                               names=['Review', 'Sentiment'])
    imdb_test = pd.read_table(r'./Datasets/imdb-test.txt', sep='\t', lineterminator='\n', header=None,
                              names=['Review', 'Sentiment'])

    # Aggregate the datasets for easier processing
    yelp = {'train': yelp_train, 'valid': yelp_valid, 'test': yelp_test}
    imdb = {'train': imdb_train, 'valid': imdb_valid, 'test': imdb_test}

    # Group sets
    training = {'yelp': yelp_train, 'imdb': imdb_train}
    valid = {'yelp': yelp_valid, 'imdb': imdb_valid}
    test = {'yelp': yelp_test, 'imdb': imdb_test}

    # Sanity Check
    for name, training_set in training.items():
        print(name.upper(), 'size:', str(len(training_set)))
        print(training_set.head(), '\n', '-' * 50)

    '''
    Question 1: Prepare the data
    '''
    # Preprocess the training sets:
    for training_set in training.values():
        preprocess_text(training_set, "Review")

    # Verify preprocessing
    for name, training_set in training.items():
        print(name.upper(), 'size:', str(len(training_set)))
        print(training_set.head(), '\n', '-' * 50)

    # Get the frequencies in descending order
    # frequency_yelp = frequency_count('', False)
    print(yelp_train['Review'].str.split())
