import csv
import nltk
import string
import re
import math
import numpy as np
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from nltk.stem.snowball import SnowballStemmer
import utilities


# The Snowball Stemmer is a stemming algorithm developed by Martin Porter. Stemming is the process of reducing words to their base or root form, which helps in information retrieval and text analysis tasks.
stemmer = SnowballStemmer('english')
bbc_dataset = "./data/BBC/bbc-text.csv"
stopwords_file_path = "./data/Stopwords/stopwords.csv"

stopwords = utilities.get_stemmed_stopwords_from_cvs(stopwords_file_path)
data = utilities.read_in_csv(bbc_dataset)
# print(data)
data_dict = utilities.get_cvs_data_as_dictionary(bbc_dataset)
for topic in data_dict.keys():
    print(topic, "\t", len(data_dict[topic]))
business_data = data_dict["business"]
sports_data = data_dict["sport"]
business_string = " ".join(business_data)
sports_string = " ".join(sports_data)
# print(business_string)
freq_dist = utilities.get_frequency_distribution_for_text(
    business_string, stopwords)
# print(freq_dist.most_common(200))
(business_train_data, business_test_data) = utilities.split_test_train(business_data, 0.8)
(sports_train_data, sports_test_data) = utilities.split_test_train(sports_data, 0.8)
train_data = business_train_data + sports_train_data
(tfidf_vectorizer, tfidf_matrix) = utilities.create_tfid_vectorizer_and_matrix(
    train_data, stopwords)
le = utilities.get_labels(["business", "sport"])
print(le)
classes = le.classes_
print(f"Label Classes {classes}")
train_data_dict = {'business': business_train_data, 'sport': sports_train_data}
test_data_dict = {'business': business_test_data, 'sport': sports_test_data}
(X_train, y_train_labels) = \
    utilities.create_dataset(tfidf_vectorizer, train_data_dict, classes, le)
(X_test, y_test_labels) = \
    utilities.create_dataset(tfidf_vectorizer, test_data_dict, classes, le)
#utilities.show_vector_matrix(tfidf_vectorizer, tfidf_matrix)
utilities.predict_trivial(X_train, y_train_labels, X_test, y_test_labels, le)
# get_stats(business_string)
# get_stats(sports_string)
