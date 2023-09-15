import numpy as np
import string
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from itertools import repeat
from nltk.probability import FreqDist
import utilities

bbc_dataset = "./data/BBC/bbc-text.csv"
stopwords_file_path = "./data/Stopwords/stopwords.csv"


def evaluate(X, y, le):
    y_pred = np.array(list(map(utilities.classify_vector, X,
                               )))
    print(classification_report(y, y_pred,
                                labels=le.transform(le.classes_),
                                target_names=le.classes_))


def main():
    data_dict = utilities.get_cvs_data_as_dictionary(bbc_dataset)
    stopwords = utilities.get_stemmed_stopwords_from_cvs(stopwords_file_path)
    (train_dict, test_dict) = utilities.divide_data_using_dictionary_keys(data_dict)
    le = utilities.get_labels(list(data_dict.keys()))
    vectorizer_dict = utilities.create_vectorizers_dictionary_of_most_common_words(
        train_dict, stopwords)
    (X, y) = utilities.create_dataset_from_dictionary(
        test_dict, le, vectorizer_dict)
    evaluate(X, y, le)


if (__name__ == "__main__"):
    main()
