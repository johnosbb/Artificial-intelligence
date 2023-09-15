import numpy as np
import string
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from itertools import repeat
from nltk.probability import FreqDist
import utilities


business_vocabulary = ["market", "company", "growth", "firm", "economy", "government", "bank",
                       "sales", "oil", "prices", "business", "uk", "financial", "dollar", "stock",
                       "trade", "investment", "quarter", "profit", "jobs", "foreign", "tax",
                       "euro", "budget", "cost", "money", "investor", "industry", "million", "debt"]

sports_vocabulary = ["game", "england", "win", "player", "cup", "team", "club", "match",
                     "set", "final", "coach", "season", "injury", "victory", "league", "play",
                     "champion", "olympic", "title", "ball", "sport", "race", "football", "rugby",
                     "tennis", "basketball", "hockey"]


# It converts a collection of text documents into a matrix of token counts.
business_vectorizer = CountVectorizer(vocabulary=business_vocabulary)
sports_vectorizer = CountVectorizer(vocabulary=sports_vocabulary)

bbc_dataset = "./data/BBC/bbc-text.csv"
stopwords_file_path = "./data/Stopwords/stopwords.csv"

# we take in a particular article of type sport or business
# we create a matrix using the vectorizer we have previously created with the feature vocabulary
# we then see how many hits we have in the matrix for the feature words.


def transform(text):
    # Extract token counts out of raw text documents using the vocabulary fitted with fit or the one provided to the constructor.
    business_X = business_vectorizer.transform([text])  # these create a matrix
    sports_X = sports_vectorizer.transform([text])
    #utilities.show_vector_matrix(business_vectorizer, business_X)
    # sum = calculate the sum of elements in an iterable, such as a list, tuple
    # creates a dense matrix, which we convert to a list
    business_X_as_list = business_X.todense().tolist()
    # print(business_X_as_list)
    #print(f"First row only\n {business_X_as_list[0]}")
    # This list will only have one row, thus [0]
    business_sum = sum(business_X_as_list[0])
    sports_X_as_list = sports_X.todense().tolist()
    sports_sum = sum(sports_X_as_list[0])
    return np.array([business_sum, sports_sum])


# takes the BBC dataset and the label classes
def create_dataset_as_np_array(data_dict, le):
    data_matrix = []
    labels = le.classes_
    classifications = []
    gold_labels = []
    for label in labels:
        for text in data_dict[label]:
            gold_labels.append(le.transform([label]))
            # The text vector is the combined sum of hits for the sports and business vocabulary
            text_vector = transform(text)
            data_matrix.append(text_vector)
    X = np.array(data_matrix)
    y = np.array(gold_labels)
    return (X, y)

# This will receive a vector which is a single row consisting of two columns
# The first column represents the number of business hits
# The second is the number of sport hits


def classify(vector, le):
    label = ""
    if (vector[0] > vector[1]):
        label = "business"
    else:
        label = "sport"
    return le.transform([label])


def evaluate(X, y, le):
    # The repeat(le) is an iterable that repeats the le object (assumed to be a LabelEncoder) indefinitely.
    y_pred = np.array(list(map(classify, X, repeat(le))))
    print(classification_report(y, y_pred, labels=le.transform(
        le.classes_), target_names=le.classes_))


def main():
    le = utilities.get_labels(["business", "sport"])
    data_dict = utilities.get_cvs_data_as_dictionary(bbc_dataset)
    (X, y) = create_dataset_as_np_array(data_dict, le)
    print(f"X shape (rows,columns):{X.shape}, y shape {y.shape}")
    evaluate(X, y, le)


if (__name__ == "__main__"):
    # auto()
    main()
