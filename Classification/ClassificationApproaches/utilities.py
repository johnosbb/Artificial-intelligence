import time
import nltk
import spacy
import string
import Levenshtein
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np
from nltk import FreqDist
import gensim
import pickle
from os import listdir
from os.path import isfile, join
import json
import pandas as pd
from langdetect import detect
import re
import math
from sklearn import preprocessing
from sklearn import svm
from spacy.tokens import DocBin
from tqdm import tqdm

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
nlp = spacy.load("en_core_web_sm")
stemmer = SnowballStemmer('english')


def get_top3_similar_words(model, search_term):
    similarity_list = model.most_similar(search_term, topn=3)
    similar_words = [sim_tuple[0] for sim_tuple in similarity_list]
    return similar_words


def train_word2vec(words, word2vec_model_path):
    # model = gensim.models.Word2Vec(
    #    words,
    #    size=50,
    #    window=7,
    #    min_count=1,
    #    workers=10)
    model = gensim.models.Word2Vec(words, window=5, size=200, min_count=5)
    model.train(words, total_examples=len(words), epochs=200)
    pickle.dump(model, open(word2vec_model_path, 'wb'))
    return model


def create_and_save_word2vec_model(words, filename):
    model = gensim.models.Word2Vec(words, min_count=1)
    model.train(words, total_examples=model.corpus_count, epochs=400)
    pickle.dump(model, open(filename, 'wb'))
    return model


def evaluate_model(model, filename):
    return model.wv.evaluate_word_analogies(filename)


def get_phrases(text, stopwords):
    words = nltk.tokenize.word_tokenize(text)
    phrases = {}
    current_phrase = []
    for word in words:
        if (word in stopwords or word in string.punctuation):
            if (len(current_phrase) > 1):
                phrases[" ".join(current_phrase)] = "_".join(current_phrase)
                current_phrase = []
        else:
            current_phrase.append(word)
    if (len(current_phrase) > 1):
        phrases[" ".join(current_phrase)] = "_".join(current_phrase)
    return phrases


def tokenize_nltk(text):
    return nltk.tokenize.word_tokenize(text)


def get_all_book_sentences(directory):
    text_files = [join(directory, f) for f in listdir(
        directory) if isfile(join(directory, f)) and ".txt" in f]
    all_sentences = []
    for text_file in text_files:
        sentences = get_sentences(text_file)
        all_sentences = all_sentences + sentences
    return all_sentences

# Model Training Parameters:
# min_count is the minimum number of times a word
# has to occur in the training corpus, with the default being 5. The size parameter sets the
# size of the word vector. window restricts the maximum number of words between the
# predicted and current word in a sentence. workers is the number of working threads;
# the more there are, the quicker the training will proceed. When training the model, the
# epoch parameter will determine the number of training iterations the model will
# go through.


def train_word2vec(words, word2vec_model_path):
    # model = gensim.models.Word2Vec(
    #    words,
    #    size=50,
    #    window=7,
    #    min_count=1,
    #    workers=10)
    model = gensim.models.Word2Vec(
        words, window=5, vector_size=200, min_count=1)
    model.train(words, total_examples=len(words), epochs=200)
    pickle.dump(model, open(word2vec_model_path, 'wb'))
    return model

# a dense matrix is a matrix representation where all elements are explicitly stored in memory,
# taking up space proportional to the number of elements in the matrix.
# Each element of the matrix is represented by a fixed-size data type, such as a floating-point number.
# Contrastingly, a sparse matrix is a matrix representation optimized for matrices that contain mostly zero elements.
# In a sparse matrix, only the non-zero elements are stored along with their indices, resulting in efficient memory usage.
# This can be particularly useful when dealing with large matrices where most elements are zero.

# This function takes in a list of texts, the vectorizer, the label being used, and the
# label encoder. It then creates the vector representation of the text using the vectorizer. It
# also creates a list of labels and then encodes them using the label encoder


def create_data_matrix(input_data, vectorizer, label, le):
    # transformation to the input_data, converting the text documents into numerical feature vectors. The resulting vectors are converted from a sparse matrix representation to a dense matrix
    vectors = vectorizer.transform(input_data).todense()
    # This line creates a list of labels by repeating the label value for the number of documents in the input_data. This ensures that each document is associated with the same label.
    labels = [label]*len(input_data)
    # Transform the list of labels into their corresponding numeric representations.
    enc_labels = le.transform(labels)
    return (vectors, enc_labels)


def create_dataset(vectorizer, data_dict, labels, le):
    # business_news = data_dict["business"]
    # sports_news = data_dict["sport"]
    category_vectors = []
    category_labels = []
    for label in labels:
        category = data_dict[label]
        (category_vector, category_label) = create_data_matrix(
            category, vectorizer, label, le)
        category_vectors.append(category_vector)
        category_labels.append(category_label)
    vector_tuple = tuple(category_vectors)
    # (sports_vectors, sports_labels) = create_data_matrix(
    #     sports_news, vectorizer, "sport", le)
    # (business_vectors, business_labels) = create_data_matrix(
    #     business_news, vectorizer, "business", le)
    all_data_matrix = np.vstack((vector_tuple))
    labels_concat = np.concatenate(category_labels)
    return (all_data_matrix, labels_concat)


def predict_trivial(X_train, y_train, X_test, y_test, le):
    # The random_state parameter is set to 0 to ensure reproducibility,
    # so that we get the same random predictions each time we run the code.
    # This strategy generates random predictions by uniformly selecting classes. It does not consider the input features and predicts each class with equal probability.
    dummy_clf = DummyClassifier(strategy='uniform', random_state=0)
    dummy_clf.fit(X_train, y_train)  # The fit method trains the classifier using the input features X_train and the corresponding target labels y_train. In this case, since the 'uniform' strategy doesn't consider the input features, the training process doesn't involve any actual learning. The purpose is simply to prepare the classifier for making predictions based on the specified strategy.
    # In this case, the predictions are randomly selected classes based on the 'uniform' strategy. The predicted labels are assigned to the variable y_pred.
    y_pred = dummy_clf.predict(X_test)
    print(dummy_clf.score(X_test, y_test))
    print(classification_report(y_test, y_pred, labels=le.transform(
        le.classes_), target_names=le.classes_))


def show_vector_matrix(vectorizer, matrix):
    # Convert sparse matrix to COO format
    matrix_coo = matrix.tocoo()
    feature_names = vectorizer.get_feature_names_out()
    densematrix = matrix.todense()
    print(f"Dense Matrix: {densematrix}")
    print(f"Sparse Matrix: \n")
    for row, col, value in zip(matrix_coo.row, matrix_coo.col, matrix_coo.data):
        print(
            f"Row: {row}, Index: {col}, Value: {value} Term: {feature_names[col]}")


def create_and_save_frequency_dist(word_list, filename):
    fdist = FreqDist(word_list)
    pickle.dump(fdist, open(filename, 'wb'))
    return fdist


def read_in_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
    return data_read


def replace_phrases(phrases_dict, text):
    for phrase in phrases_dict.keys():
        text = text.replace(phrase, phrases_dict[phrase])
    return text


def read_text_file(filename):
    file = open(filename, "r", encoding="utf-8")
    return file.read()


def write_text_to_file(text, filename):
    text_file = open(filename, "w", encoding="utf-8")
    text_file.write(text)
    text_file.close()


def get_json_as_text(filename, chunkSize=10000):
    reader = pd.read_json(filename, orient="records",
                          lines=True, chunksize=chunkSize)
    chunk = next(reader)
    text = ''
    for index, row in chunk.iterrows():
        row_text = row['text']
        lang = detect(row_text)
        if (lang == "en"):
            text = text + row_text.lower()
    return text


def preprocess_text(text):
    text = text.replace("\n", " ")
    return text


def get_frequency_distribution_for_text(text, stopwords, num_words=200):
    word_list = tokenize_nltk(text)
    word_list = [word for word in word_list if word not in stopwords and re.search(
        "[A-Za-z]", word)]
    freq_dist = FreqDist(word_list)
    return freq_dist


def divide_into_sentences_nltk(text):
    sentences = tokenizer.tokenize(text)
    return sentences

# Out-of-vocabulary words: Character n-gram models are more robust when dealing with out-of-vocabulary words, i.e.,
# words that are not present in the training data.
# Since character n-grams operate on individual characters,
# they can still capture useful information from unseen words by considering their character-level patterns.
# In contrast, word-based models typically struggle
# with out-of-vocabulary words because they treat each word as a discrete unit.

# Morphologically rich languages: Character n-gram models are particularly useful for morphologically rich languages,
# where words undergo various inflections and morphological changes.
# By considering character-level information, these models can capture the underlying morphological
# structure and handle word forms more effectively.
# In such cases, word-based models may struggle due to the increased vocabulary size and sparsity.

# Misspellings and noisy text: Character n-gram models can handle misspelled words and noisy text better
# than word-based models. By considering the character-level patterns,
# these models can identify similarities between correct and misspelled versions of words,
# improving their ability to recognize and correct misspellings

# Short and contextually ambiguous text: In scenarios where the input text is short or lacks sufficient context,
# character n-gram models can often provide more reliable results.
# By focusing on character-level patterns, these models can leverage smaller units of information
# and make predictions based on the available characters, even when the overall context is limited.

# This function will return a char vector and a matrix


def create_char_vectorizer_tfid(sentences):
    # Create TF-IDF object
    tfidf_char_vectorizer = TfidfVectorizer(analyzer='char_wb', max_df=0.90, max_features=200000,
                                            min_df=0.05, use_idf=True, ngram_range=(1, 3))
    tfidf_char_vectorizer = tfidf_char_vectorizer.fit(sentences)
    tfidf_matrix = tfidf_char_vectorizer.transform(sentences)
    return (tfidf_char_vectorizer, tfidf_matrix)


# Word frequencies are calculated as follows. For each word, the overall
# frequency is a product of the term frequency and the inverse document frequency. Term
# frequency is the number of times the word occurs in the document. Inverse document
# frequency is the total number of documents divided by the number of documents where
# the word occurs.

# For some words like “happy” and “happiness” the stemming process will convert these to the same stem “happi”.
# These are grouped together as one feature.
# Stemming can convert word features to stem features,
# which is effective in reducing the size of features.
# However, there are some problems:
# The stemmer rules are manually crafted based on statistics,
# so it’s not always correct when given a large sample vocabulary (Porter, 2001).
# Stems could be meaningless words that are not in dictionaries. (e.g. “is” -> “i”, “happy”->”happi”)

# This function assumes stopwords have already been stemmed
# It creates a vectorizer and a matrix
# The vectorizer can be used later for transforming new sentences into numerical feature vectors
# using the learned vocabulary and IDF weights,
# while the TF-IDF matrix represents the numerical representation of the input sentences suitable
# for machine learning tasks.
def create_tfid_vectorizer_and_matrix(sentences, stopword_list):
    # Create TF-IDF object
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=200000,
                                       min_df=0.05, stop_words=stopword_list,
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    tfidf_vectorizer = tfidf_vectorizer.fit(sentences)
    tfidf_matrix = tfidf_vectorizer.transform(sentences)
    return (tfidf_vectorizer, tfidf_matrix)


# This function creates the vectorizer only

def create_tfid_vectorizer(data, stopword_list):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=0.05, stop_words=stopword_list,
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    tfidf_vectorizer.fit(data)
    return tfidf_vectorizer


def make_predictions(test_data, vectorizer, km):
    predicted_data = {}
    for topic in test_data.keys():
        this_topic_list = test_data[topic]
        if (topic not in predicted_data.keys()):
            predicted_data[topic] = {}
        for text in this_topic_list:
            prediction = km.predict(vectorizer.transform([text]))[0]
            if (prediction not in predicted_data[topic].keys()):
                predicted_data[topic][prediction] = []
            predicted_data[topic][prediction].append(text)
    return predicted_data


def clean_text(text):
    for segment in text:
        # print (segment)
        segment = segment.strip()
        segment = segment.replace("\n", " ")  # remove line breaks
        # print(segment)

        punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''  # punctuation
        for ele in segment:
            if ele in punctuation:
                # remove  punctuation replacing with an empty string
                segment = segment.replace(ele, "")
        # print (segment)
        words = segment.split()

# create a vectorizer of the most popular  words in each category of the dictionary provided
# the dictionary has the structure "topic" : list of strings with each string containing a text article


def create_vectorizers_dictionary_of_most_common_words(data_dict, stopwords):
    topic_list = list(data_dict.keys())
    vectorizer_dict = {}
    for topic in topic_list:
        # a list of the articles for a given topic
        text_array = data_dict[topic]
        # concatanate all articles into one big string
        text = " ".join(text_array)
        word_list = tokenize_nltk(text)  # tokenize the string
        # exclude stopwords
        word_list = [word for word in word_list if word not in stopwords]
        freq_dist = FreqDist(word_list)
        # the most frequently occurring words for that topic.
        top_200 = freq_dist.most_common(200)
        vocab = [wtuple[0] for wtuple in top_200 if wtuple[0]
                 not in stopwords and wtuple[0] not in string.punctuation]
        # Convert a collection of text documents to a matrix of token counts.
        vectorizer_dict[topic] = CountVectorizer(vocabulary=vocab)
    return vectorizer_dict


def get_most_frequent_words(text, stopwords, number_of_words):
    word_list = tokenize_nltk(text)
    word_list = [word for word in word_list if word not in stopwords and word not in string.punctuation and re.search(
        '[a-zA-Z]', word)]
    freq_dist = FreqDist(word_list)
    top_n = freq_dist.most_common(number_of_words)
    top_n = [word[0] for word in top_n]
    return top_n


def show_count_vector_features(count_vector):
    print(count_vector.get_feature_names())


def get_labels(names):
    # class in scikit-learn is used for encoding categorical labels into numeric representations.
    le = preprocessing.LabelEncoder()
    # This method is used to fit the encoder to the given labels. It takes a list or array-like object labels as input, representing the categorical labels that need to be encoded. During the fitting process, the encoder learns the unique classes or categories present in the labels.
    le.fit(names)
    return le


def get_w2vec_word_vectors(sentence, model):
    word_vectors = []
    for word in sentence:
        try:
            word_vector = model.get_vector(word.lower())
            word_vectors.append(word_vector)
        except KeyError:
            continue
    return word_vectors

# In the function below we average the word vectors. Averaging the word vectors to get the sentence vector is only one way of
# approaching this task, and is not without its problems. One alternative is to
# train a doc2vec model, where sentences, paragraphs, and whole documents can
# all be units instead of words.


def get_w2vec_sentence_vector(word_vectors):
    matrix = np.array(word_vectors)
    centroid = np.mean(matrix[:, :], axis=0)
    return centroid  # returns the average vector length

# Split data into two sets, one for training and one for testing


def split_test_train(data, train_percent):
    train_test_border = math.ceil(train_percent*len(data))
    train_data = data[0:train_test_border]
    test_data = data[train_test_border:]
    return (train_data, test_data)


def divide_data_using_dictionary_keys(data_dict):
    train_dict = {}
    test_dict = {}
    for topic in data_dict.keys():
        text_list = data_dict[topic]
        # Split arrays or matrices into random train and test subsets.
        x_train, x_test = train_test_split(text_list, test_size=0.2)
        train_dict[topic] = x_train
        test_dict[topic] = x_test
    return (train_dict, test_dict)


def transform_vectorizing_dictionary(text, vect_dict, le):
    number_topics = len(list(vect_dict.keys()))
    sum_list = [0]*number_topics
    for topic in vect_dict.keys():
        vectorizer = vect_dict[topic]
        this_topic_matrix = vectorizer.transform([text])
        this_topic_sum = sum(this_topic_matrix.todense().tolist()[0])
        index = le.transform([topic])[0]
        sum_list[index] = this_topic_sum
    return np.array(sum_list)


# takes a test data set and a vectorizer of the most common words in various classes
def create_dataset_from_dictionary(data_dict, le, vectorizer_dict):
    data_matrix = []
    gold_labels = []
    for topic in data_dict.keys():
        for text in data_dict[topic]:
            gold_labels.append(le.transform([topic]))
            text_vector = transform_vectorizing_dictionary(
                text, vectorizer_dict, le)
            data_matrix.append(text_vector)
    X = np.array(data_matrix)
    y = np.array(gold_labels)
    return (X, y)


def create_dataset_as_pandas_frame(data_dict, le):
    text = []
    labels = []
    for topic in data_dict:
        label = le.transform([topic])
        text = text + data_dict[topic]
        this_topic_labels = [label[0]]*len(data_dict[topic])
        labels = labels + this_topic_labels
    docs = {'text': text, 'label': labels}
    frame = pd.DataFrame(docs)
    return frame


# X_train: This variable contains the training data for the input features (also known as the independent variables). It is a subset of the original DataFrame's train_column_name column. The machine learning model will be trained using this data.
# X_test: This variable contains the testing data for the input features. It is also a subset of the original DataFrame's train_column_name column, but it contains different data points than X_train. The machine learning model will be evaluated on this data to measure its performance.
# y_train: This variable contains the training data for the target variable (also known as the dependent variable or labels). It is a subset of the original DataFrame's gold_column_name column, corresponding to the data used in X_train.
# y_test: This variable contains the testing data for the target variable. Similar to X_test, it is a subset of the original DataFrame's gold_column_name column, corresponding to the data used in X_test. The model's predictions on this data will be compared to the actual labels (y_test) to evaluate its performance.
def split_pandas_frame_dataset(df, train_column_name, gold_column_name, test_percent):
    X_train, X_test, y_train, y_test = train_test_split(
        df[train_column_name], df[gold_column_name], test_size=test_percent, random_state=0)
    return (X_train, X_test, y_train, y_test)


def classify_vector(vector):
    # np.where returns elements chosen from x or y depending on condition. amax returns the maximum of an array or maximum along an axis
    result = np.where(vector == np.amax(vector))
    label = result[0][0]
    return [label]


def train_svm_classifier(X_train, y_train):
    clf = svm.SVC(C=1, kernel='linear')
    clf = clf.fit(X_train, y_train)
    return clf

# We read in a cvs file of stopwords and then stem them returning a stemmed list.


def get_stemmed_stopwords_from_cvs(path):
    stopwords = read_in_csv(path)
    stopwords = [word[0] for word in stopwords]
    stemmed_stopwords = [stemmer.stem(word) for word in stopwords]
    stopwords = stopwords + stemmed_stopwords
    return stopwords

# We read in a cvs file of stopwords and then return them as is without stemming


def get_stopwords_from_cvs(path):
    stopwords = read_in_csv(path)
    stopwords = [word[0] for word in stopwords]
    return stopwords


def load_w2vec_model(w2vec_model_path):
    model = KeyedVectors.load_word2vec_format(w2vec_model_path, binary=True)
    return model


def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in string.punctuation]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def divide_into_sentences_spacy(text):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


def divide_into_sentences(text):
    return divide_into_sentences_nltk(text)


def get_sentences(filename):
    text = read_text_file(filename)
    text = preprocess_text(text)
    sentences = divide_into_sentences_nltk(text)
    return sentences


def get_cvs_data_as_dictionary(filename):
    data = read_in_csv(filename)
    data_dict = {}
    for row in data[1:]:
        category = row[0]
        text = row[1]
        if (category not in data_dict.keys()):
            data_dict[category] = []
        data_dict[category].append(text)
    return data_dict


# Get items from a dataframe
# Assumes DF has a column called "Job Description"
def get_items_from_dataframe(df, regex, column_name, target_column_name):
    df[column_name] = df[target_column_name].apply(
        lambda x: re.findall(regex, x))
    return df


# Get a list of items from a dataframe
def get_list_of_items_from_dataframe(df, column_name):
    items = []
    for index, row in df.iterrows():
        if (len(row[column_name]) > 0):
            for item in list(row[column_name]):
                if (type(item) is tuple and len(item) > 1):
                    item = item[0]
                if (item not in items):
                    items.append(item)
    return items


# Get a list of emails from a dataframe
# Assumes DF has a column called "Job Description"
def get_emails_from_dataframe(df, target_column_name):
    original_email_regex = '[\S]+@[a-zA-Z0-9\.]+\.[a-zA-Z]+'
    email_regex = '[^\s:|()\']+@[a-zA-Z0-9\.]+\.[a-zA-Z]+'
    df['emails'] = df[target_column_name].apply(
        lambda x: re.findall(email_regex, x))
    emails = get_list_of_items_from_dataframe(df, 'emails')
    return emails

# Get a list of urls from a dataframe


def get_urls_from_dataframe(df, target_column_name):
    url_regex = '(http[s]?://(www\.)?[A-Za-z0-9–_\.\-]+\.[A-Za-z]+/?[A-Za-z0-9$\–_\-\/\.]*)[\.)\"]*'
    df = get_items_from_dataframe(df, url_regex, 'urls', target_column_name)
    urls = get_list_of_items_from_dataframe(df, 'urls')
    return urls


# The Levenshtein distance, also known as the edit distance,
# is a metric used to measure the similarity or dissimilarity between two strings or sequences.
# It quantifies the minimum number of single-character edits (insertions, deletions, or substitutions)
# required to transform one string into another.
# In the context of similarity, a lower Levenshtein distance implies greater similarity between the two strings.

# How it works:
# Insertion: Adding a character to one of the strings.
# Example: "kitten" and "kittens" have a Levenshtein distance of 1 because you need to insert an 's' to make them the same.
# Deletion: Removing a character from one of the strings.
# Example: "flaw" and "law" have a Levenshtein distance of 1 because you need to delete the 'f' from the first string to make them the same.
# Substitution: Replacing a character in one of the strings with another character.
# Example: "cat" and "hat" have a Levenshtein distance of 1 because you need to substitute 'c' with 'h' to make them the same.
# The Levenshtein distance can be useful in various applications, including spell-checking, DNA sequence alignment, and natural language processing. It provides a way to quantify how different two strings are, which is often used to determine the similarity or dissimilarity between words or phrases in a text analysis context. The smaller the Levenshtein distance between two strings, the more similar they are considered to be.
# .apply(lambda x: Levenshtein.distance(input_string, x)):
# This is using the apply method to apply a lambda function to each element (x) in the 'emails' column.
# The lambda function calculates the Levenshtein distance between input_string (the string you want to compare against)
# and each element x (an email address from the 'emails' column).


def find_levenshtein_dataframe(input_string, df, column_name):
    df['distance_to_' + input_string] = df[column_name].apply(
        lambda x: Levenshtein.distance(input_string, x))
    return df


def find_jaro_dataframe(input_string, df, column_name):
    df['distance_to_' + input_string] = df[column_name].apply(
        lambda x: Levenshtein.jaro(input_string, x))
    return df

# add an ner component to a spacy model


# returns an ner pipline, creating it if it does not exist
def add_ner_to_model(nlp):
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    return (nlp, ner)


# expects data in spacy training format
#             "<referenced text>",
#             {
#                 "entities": [
#                     [
#                         35,
#                         55,
#                         "<LABEL_NAME>"
#                     ]
#                 ]
#             }
def add_labels_to_ner_model(ner, data):
    for sentence, annotations in data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])  # label is the 3rd field of entities
    return ner


def convert_ner_data_to_spacy_format(data_path, target_directory, use_blank_model=True):
    if(use_blank_model):
        nlp = spacy.blank("en")  # load a new blank spacy model
    else:
        nlp = spacy.load("en_core_web_sm")  # load default web model
    db = DocBin()  # create a DocBin object
    with open(data_path, 'r') as f:
        data = json.load(f)
    train_data = data['annotations']
    train_data = [tuple(i) for i in train_data]
    for text, annot in tqdm(train_data):  # data in previous format
        doc = nlp.make_doc(text)  # create doc object from text
        ents = []
        for start, end, label in annot["entities"]:  # add character indexes
            span = doc.char_span(start, end, label=label,
                                 alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents  # label the text with the ents
        db.add(doc)
    db.to_disk(target_directory)  # save the docbin object
