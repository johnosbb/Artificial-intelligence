import numpy as np
import pandas as pd
import re
import nltk
import email
import os
import csv
import pickle

nltk.download('stopwords')
from nltk.corpus import stopwords
# Assuming stopwords is a WordListCorpusReader
stopwords_list = stopwords.words('english')


Nsamp = 1000 # number of samples to generate in each class - 'spam', 'not spam'
maxtokens = 200 # the maximum number of tokens per document
maxtokenlen = 100 # the maximum length of each token
filepath = "./data/spam_detection/emails.csv"


emails = pd.read_csv(filepath)
 
print("Successfully loaded {} rows and {} columns!".format(emails.shape[0],  emails.shape[1]))
print(emails.head(n=5))
print(emails.loc[0]["message"])


def extract_messages(df):
    messages = []
    for item in df["message"]:
        e = email.message_from_string(item)
        message_body = e.get_payload()
        messages.append(message_body)
    print("Successfully retrieved message body from emails!")
    return messages


# The `assemble_bag()` function assembles a new dataframe containing all the unique words found in the text documents.
# It counts the word frequency and then returns the new dataframe.
def assemble_bag(data):
    used_tokens = [] #  A list to keep track of tokens that have already been encountered and added to the DataFrame.
    all_tokens = [] # A list to store all unique tokens encountered in the input data.
 
    # Nested loops go through each item in the input data and each token in each item.
    # If a token is already in all_tokens, it checks whether it has been added to used_tokens yet.
    # If not, it adds it to used_tokens. 
    # If the token is not in all_tokens, it adds it to all_tokens.
    for item in data:
        for token in item:
            if token in all_tokens:
                if token not in used_tokens:
                    used_tokens.append(token)
            else:
                all_tokens.append(token)
    # Creates a DataFrame (df) with a row for each item in the input data and columns for each token in used_tokens.
    # The DataFrame is initialized with zeros.
    df = pd.DataFrame(0, index = np.arange(len(data)), columns = used_tokens)
    # Iterates over each item in the input data and each token in each item.
    # If the token is in used_tokens, it increments the corresponding count in the DataFrame.
    for i, item in enumerate(data):
        for token in item:
            if token in used_tokens:
                df.iloc[i][token] += 1    
    return df


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

    #print(stopwords_list)
    token = [token for token in row if token not in stopwords_list]
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
            text = reg_expressions(text) # should this not limit the length?
            data.append(text)
            sentiments.append(sentiment)
            # Find the maximum length of sublists
    max_length = max(len(seq) for seq in data)
    # Pad sequences with zeros to make them of equal length
    data_padded = [seq + [0] * (max_length - len(seq)) for seq in data] 
    data_np = np.array(data_padded)
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

# RAw data is a 2-dimensional array (or matrix) containing the raw data.
# Each row represents a data point, and each column represents a feature.
# Iterate over each row (i) in the raw_data array.
# Join the elements of the row (raw_data[i]) into a single string using ' 
def convert_data(raw_data,header):
    converted_data, labels = [], []
    for i in range(raw_data.shape[0]):
        out = ' '.join(raw_data[i])
        converted_data.append(out)
        labels.append(header[i])
    # After the loop, convert the converted_data list to a NumPy array using np.array(converted_data, dtype=object).
    # The dtype=object ensures that each element of the array can be of 
    # variable length (since strings can have different lengths).
    # Reshape the array using [:, np.newaxis]. 
    # This adds a new axis to the array, effectively converting each string element into a one-element subarray.
    converted_data = np.array(converted_data, dtype=object)[:, np.newaxis]        
    return converted_data, np.array(labels)
 
train_path = os.path.join('./data/movie_database/aclImdb', 'train')
raw_data, raw_header = load_data(train_path)

print(raw_data.shape)
print(len(raw_header))

random_indices = np.random.choice(range(len(raw_header)),size=(Nsamp*2,),replace=False)
data_train = raw_data[random_indices]
header = raw_header[random_indices]

# we need to check the balance of the resulting data with regard to class. 
# In general, we don’t want one of the labels to represent most of the dataset, 
# unless that is the distribution expected in practice.
unique_elements, counts_elements = np.unique(header, return_counts=True)
print("Sentiments and their frequencies:")
print(unique_elements)
print(counts_elements)


bodies = extract_messages(emails)
bodies_df = pd.DataFrame(bodies)
print(bodies_df.head(n=5))

filepath = "./data/spam_detection/fradulent_emails.txt"
with open(filepath, 'r',encoding="latin1") as file:
    data = file.read()
    
fraud_emails = data.split("From r")
 
print("Successfully loaded {} spam emails!".format(len(fraud_emails)))

fraud_bodies = extract_messages(pd.DataFrame(fraud_emails,columns=["message"],dtype=str))
fraud_bodies_df = pd.DataFrame(fraud_bodies[1:])
print(fraud_bodies_df.head())
stopwords = stopwords.words('english')    
 
EnronEmails = bodies_df.iloc[:,0].apply(tokenize)
EnronEmails = EnronEmails.apply(stop_word_removal)
EnronEmails = EnronEmails.apply(reg_expressions)
EnronEmails = EnronEmails.sample(Nsamp)
 
SpamEmails = fraud_bodies_df.iloc[:,0].apply(tokenize)
SpamEmails = SpamEmails.apply(stop_word_removal)
SpamEmails = SpamEmails.apply(reg_expressions)
SpamEmails = SpamEmails.sample(Nsamp)
 

raw_data = pd.concat([SpamEmails,EnronEmails], axis=0).values

# Convert the NumPy array to a Pandas DataFrame
raw_data_df = pd.concat([SpamEmails, EnronEmails], axis=0)

print("Shape of combined data is:")
print(raw_data.shape)
print("Data is:")
print(raw_data)

# create corresponding labels
Categories = ['spam','notspam']
header = ([1]*Nsamp)
header.extend(([0]*Nsamp))

# create bag-of-words model
EnronSpamBag = assemble_bag(raw_data)
# this is the list of words in our bag-of-words model
predictors = [column for column in EnronSpamBag.columns]

# Define the file name
csv_file = "./data/spam_detection/predictors.csv"

# Write 'predictors' list to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(predictors)
    
   
    


# we will need to adapt our preprocessed data for this model architecture.
# The bag-of-words representation for the traditional models from the variable raw_data is a
# NumPy array containing a list of word tokens per email.
# In this case, we need to combine each such list into a single text string.
# This is the format in which the ELMo TensorFlow Hub model expects the input.

raw_data, header = unison_shuffle_data(raw_data, header)
 
idx = int(0.7*data_train.shape[0])
train_x, train_y = convert_data(raw_data[:idx],header[:idx])
test_x, test_y = convert_data(raw_data[idx:],header[idx:])

print("train_x/train_y list details, to make sure they are of the right form:")
print(len(train_x))
print(train_x)
print(train_y[:5])
print(len(train_y))

# Save datasets
with open('./data/spam_detection/train_data_elmo.pkl', 'wb') as f:
    pickle.dump((train_x, train_y), f)

with open('./data/spam_detection/test_data_elmo.pkl', 'wb') as f:
    pickle.dump((test_x, test_y), f)
