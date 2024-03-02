import numpy as np
import pandas as pd
import re
import nltk
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

maxtokens = 50
maxtokenlen = 20

def tokenize(row):
    if row in [None,'']:
        tokens = ""
    else:
        tokens = str(row).split(" ")[:maxtokens]
    return tokens

#  normalize words by turning them into lower case and removing punctuation
def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', "", token)
            token = token[:maxtokenlen]
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens

def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]
    token = filter(None, token)
    return token


def load_data(path):
    data, sentiments = [], []
    for folder, sentiment in (('neg', 0), ('pos', 1)):
        folder = os.path.join(path, folder)
        for name in os.listdir(folder):
            with open(os.path.join(folder, name), 'r') as reader:
                  text = reader.read()
            text = tokenize(text)
            text = stop_word_removal(text)
            text = reg_expressions(text)
            data.append(text)
            sentiments.append(sentiment)
    data_np = np.array(data)
    data, sentiments = unison_shuffle_data(data_np, sentiments)
    
    return data, sentiments

# Having fully vectorized the dataset, we must remember that it is not shuffled with respect to classes; 
# that is, it contains Nsamp = 1000 spam emails followed by an equal number of nonspam emails.
# Depending on how this dataset is split—in our case, by picking the first 70% for training and the 
# remainder for testing—this could lead to a training set composed of spam only, 
# which would obviously lead to failure.
# To create a randomized ordering of class samples in the dataset,
# we will need to shuffle the data in unison with the header/list of labels.
def unison_shuffle_data(data, header):
    # Generate a random permutation of indices based on the length of the header
    p = np.random.permutation(len(header))
     # Shuffle the rows of the data array using the generated permutation
    data = data[p]
    # Shuffle the elements of the header array using the same permutation
    header = np.asarray(header)[p]
    return data, header

 
train_path = os.path.join('./data/movie_database/aclImdb', 'train')
raw_data, raw_header = load_data(train_path)