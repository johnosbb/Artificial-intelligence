import utilities
import pickle
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

bbc_dataset = "./data/BBC/bbc-text.csv"
stopwords_file_path = "./data/Stopwords/stopwords.csv"
df_as_data_path = "./data/BBC/df_as_data.json"
X_df_path = "./data/BBC/X_df.json"
y_df_path = "./data/BBC/y_df.json"

# SVM (Support Vector Machine) classification is a popular supervised machine learning algorithm used for binary
# and multi-class classification tasks. The main goal of SVM is to find a hyperplane in an N-dimensional
# feature space (where N is the number of features) that best separates different classes of data points.
# Imagine you have data points in a graph, and you want to draw a line to divide them into two groups. SVM finds the line that has the largest gap between the closest data points from each group.
# These closest points are called support vectors.
# SVM can also handle situations where the data cannot be separated by a straight line.
# It can transform the data into a higher-dimensional space, where it becomes easier to find a separating plane.
# This process is called the kernel trick.
# Once SVM has learned from the data, it can predict the category of new data points based
# on their position relative to the separating line or plane.
# SVM is useful for tasks like classifying emails as spam or not spam, recognizing images of animals,
# or identifying different types of diseases based on medical data.


new_example = """iPhone 12: Apple makes jump to 5G
Apple has confirmed its iPhone 12 handsets will be its first to work on faster 5G networks. 
The company has also extended the range to include a new "Mini" model that has a smaller 5.4in screen. 
The US firm bucked a wider industry downturn by increasing its handset sales over the past year. 
But some experts say the new features give Apple its best opportunity for growth since 2014, when it revamped its line-up with the iPhone 6. 
"5G will bring a new level of performance for downloads and uploads, higher quality video streaming, more responsive gaming, 
real-time interactivity and so much more," said chief executive Tim Cook. 
There has also been a cosmetic refresh this time round, with the sides of the devices getting sharper, flatter edges. 
The higher-end iPhone 12 Pro models also get bigger screens than before and a new sensor to help with low-light photography. 
However, for the first time none of the devices will be bundled with headphones or a charger. 
Apple said the move was to help reduce its impact on the environment. "Tim Cook [has] the stage set for a super-cycle 5G product release," 
commented Dan Ives, an analyst at Wedbush Securities. 
He added that about 40% of the 950 million iPhones in use had not been upgraded in at least three-and-a-half years, presenting a "once-in-a-decade" opportunity. 
In theory, the Mini could dent Apple's earnings by encouraging the public to buy a product on which it makes a smaller profit than the other phones. 
But one expert thought that unlikely. 
"Apple successfully launched the iPhone SE in April by introducing it at a lower price point without cannibalising sales of the iPhone 11 series," noted Marta Pinto from IDC. 
"There are customers out there who want a smaller, cheaper phone, so this is a proven formula that takes into account market trends." 
The iPhone is already the bestselling smartphone brand in the UK and the second-most popular in the world in terms of market share. 
If forecasts of pent up demand are correct, it could prompt a battle between network operators, as customers become more likely to switch. 
"Networks are going to have to offer eye-wateringly attractive deals, and the way they're going to do that is on great tariffs and attractive trade-in deals," 
predicted Ben Wood from the consultancy CCS Insight. Apple typically unveils its new iPhones in September, but opted for a later date this year. 
It has not said why, but it was widely speculated to be related to disruption caused by the coronavirus pandemic. The firm's shares ended the day 2.7% lower. 
This has been linked to reports that several Chinese internet platforms opted not to carry the livestream, 
although it was still widely viewed and commented on via the social media network Sina Weibo."""


def evaluate(clf, X_test, y_test, le):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, labels=le.transform(
        le.classes_), target_names=le.classes_))


def test_new_example(input_string, clf, vectorizer, le):
    denseMatrix = vectorizer.transform([input_string]).todense()
    denseMatrixAsArray = np.asarray(vector)
    prediction = clf.predict(denseMatrixAsArray)
    print(prediction)
    label = le.inverse_transform(prediction)
    print(label)


def main():
    data_dict = utilities.get_cvs_data_as_dictionary(bbc_dataset)
    stopwords = utilities.get_stemmed_stopwords_from_cvs(stopwords_file_path)
    le = utilities.get_labels(list(data_dict.keys()))
    df = utilities.create_dataset_as_pandas_frame(data_dict, le)
    # A pandas DataFrame uses a tabular data structure to store data. It is a two-dimensional, size-mutable, and heterogeneous data structure with labeled axes (rows and columns).
    df.to_json(df_as_data_path)
    (X_train, X_test, y_train, y_test) = utilities.split_pandas_frame_dataset(
        df, 'text', 'label', 0.2)
    X_train.to_json(X_df_path)
    y_train.to_json(y_df_path)
    #tfidf_vectorizer, tfidf_matrix = utilities.create_tfid_vectorizer_and_matrix(X_train, stopwords)
    tfidf_vectorizer = utilities.create_tfid_vectorizer(X_train, stopwords)
    X_train_tfidf_sparse_matrix = tfidf_vectorizer.transform(X_train)
    X_test_tfidf_sparse_matrix = tfidf_vectorizer.transform(X_test)
    X_train_tfidf_dense_matrix = X_train_tfidf_sparse_matrix.todense()
    X_test_tfidf_dense_matrix = X_test_tfidf_sparse_matrix.todense()
    # Convert the NumPy matrix to a NumPy array
    denseTrainMatrix = np.asarray(X_train_tfidf_dense_matrix)
    denseTestMatrix = np.asarray(X_test_tfidf_dense_matrix)
    utilities.show_vector_matrix(
        tfidf_vectorizer, X_train_tfidf_sparse_matrix)
    clf = utilities.train_svm_classifier(denseTrainMatrix, y_train)
    pickle.dump(clf, open("data/BBC/bbc_svm.pkl", "wb"))
    evaluate(clf, denseTestMatrix, y_test, le)
    test_new_example(new_example, clf, tfidf_vectorizer, le)


if (__name__ == "__main__"):
    main()
