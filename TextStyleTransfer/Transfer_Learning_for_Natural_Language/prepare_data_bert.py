
import random
import numpy as np
import pandas as pd
import re
import nltk
import email
import os
import csv
import pickle
# Separate headers from the message bodies
import email
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# Params for bert model and tokenization
Nsamp = 1000 # number of samples to generate in each class - 'spam', 'not spam'
maxtokens = 200 # the maximum number of tokens per document
maxtokenlen = 100 # the maximum length of each token

def tokenize(row):
    if row is None or row == '':
        tokens = ""
    else:
        try:
            tokens = row.split(" ")[:maxtokens]
        except:
            tokens=""
    return tokens


def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', "", token)
            token = token[:maxtokenlen] # truncate token
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens


def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]
    token = filter(None, token)
    return token

def analyze_data_object(obj,description=""):
    print("------------------------Object Analysis Begins--------------------------")
    # Check if it's a Series
    if isinstance(obj, pd.Series):
        print(f"Object {obj.name} is a pandas Series")
        print(f"Object Description {description}")
        print("Shape:", obj.shape)
        print("Type of elements:", type(obj.iloc[0]))
        print("First 3 elements:")
        print(obj.head(3))
    # Check if it's a DataFrame
    elif isinstance(obj, pd.DataFrame):
        print(f"Object {obj.name} is a pandas DataFrame")
        print(f"Object Description {description}")
        print("Shape:", obj.shape)
        print("Types of elements:")
        print(obj.dtypes)
        for column in obj.columns:
            print(f"Type of elements in column '{column}':", type(obj[column].iloc[0]))
        # Check type of elements in each row
        for index, row in obj.iterrows():
            print(f"Type of elements in row '{index}':", type(row.iloc[0]))
        print("First 3 elements:")
        print(obj.head(3))
    # Check if it's a NumPy array
    elif isinstance(obj, np.ndarray):
        print("Object is a NumPy array")
        print(f"Object Description {description}")
        print("Shape:", obj.shape)
        print("Type of elements:", obj.dtype)
        print("First 3 elements:")
        print(obj[:3])
    else:
        print("Unknown type")

    print("------------------------Object Analysis Ends--------------------------")

def extract_messages(df):
    messages = []
    for item in df["message"]:
        # Return a message object structure from a string
        e = email.message_from_string(item)    
        # get message body  
        message_body = e.get_payload()
        messages.append(message_body)
    print("Successfully retrieved message body from e-mails!")
    return messages


def unison_shuffle(a, b):
    p = np.random.permutation(len(b))
    data = a[p]
    header = np.asarray(b)[p]
    return data, header

# function for converting data into the right format, due to the difference in required format from sklearn models
# we expect a single string per email here, versus a list of tokens for the sklearn models previously explored
def convert_data(raw_data,header):
    converted_data, labels = [], []
    for i in range(raw_data.shape[0]):
        out = ' '.join(raw_data[i])
        converted_data.append(out)
        labels.append(header[i])
        #print(i)
    converted_data = np.array(converted_data, dtype=object)[:, np.newaxis]
    
    return converted_data, np.array(labels)

stopwords = stopwords.words('english')    
#print(stopwords) # see default stopwords

# Input data files are available in the "../input/" directory.
filepath = "./data/spam_detection/emails.csv"

# Read the data into a pandas dataframe called emails
emails = pd.read_csv(filepath)
emails = emails.dropna() # remove any blank or invalid
print("Successfully loaded {} rows and {} columns!".format(emails.shape[0], emails.shape[1]))
print(emails.head())



# take a closer look at the first email
print(emails.loc[0]["message"])

bodies = extract_messages(emails)

bodies_df = pd.DataFrame(random.sample(bodies, 10000))

# expand default pandas display options to make emails more clearly visible when printed
pd.set_option('display.max_colwidth', 300)

bodies_df.head() # you could do print(bodies_df.head()), but Jupyter displays this nicer for pandas DataFrames

filepath = "./data/spam_detection/fradulent_emails.txt"
with open(filepath, 'r',encoding="latin1") as file:
    data = file.read()
    
# split on a code word appearing close to the beginning of each email
fraud_emails = data.split("From r")

print("Successfully loaded {} spam emails!".format(len(fraud_emails)))
fraud_bodies_raw = pd.DataFrame(fraud_emails,columns=["message"],dtype=str)
fraud_bodies_raw = fraud_bodies_raw.dropna() # remove any blank or invalid
fraud_bodies = extract_messages(fraud_bodies_raw )
fraud_bodies_df = pd.DataFrame(fraud_bodies[1:])

fraud_bodies_df.head() # you could do print(fraud_bodies_df.head()), but Jupyter displays this nicer for pandas DataFrames



# Convert everything to lower-case, truncate to maxtokens and truncate each token to maxtokenlen
EnronEmails = bodies_df.iloc[:,0].apply(tokenize)
EnronEmails = EnronEmails.apply(stop_word_removal)
EnronEmails = EnronEmails.apply(reg_expressions)
EnronEmails = EnronEmails.sample(Nsamp) # randomly sample 1000 (Nsamp emails)

SpamEmails = fraud_bodies_df.iloc[:,0].apply(tokenize)
SpamEmails = SpamEmails.apply(stop_word_removal)
SpamEmails = SpamEmails.apply(reg_expressions)
SpamEmails = SpamEmails.sample(Nsamp) # randomly sample 1000 (Nsamp emails)

raw_data = pd.concat([SpamEmails,EnronEmails], axis=0).values
analyze_data_object(raw_data,description="raw_data")
print("Shape of combined data represented as numpy array is:")
print(raw_data.shape)
print("Data represented as numpy array is:") # an array of size 2000 with one dimension
print(raw_data)

# corresponding labels
Categories = ['spam','notspam']
header = ([1]*Nsamp)
header.extend(([0]*Nsamp))

raw_data, header = unison_shuffle(raw_data, header)

# split into independent 70% training and 30% testing sets
idx = int(0.7*raw_data.shape[0])
# 70% of data for training
train_x, train_y = convert_data(raw_data[:idx],header[:idx])
# remaining 30% for testing
test_x, test_y = convert_data(raw_data[idx:],header[idx:])

print("train_x/train_y list details, to make sure it is of the right form:")
print(len(train_x))
print(train_x)
print(train_y[:5])
print(train_y.shape)
analyze_data_object(train_x,description="train_x")
analyze_data_object(train_y,description="train_y")
# Save datasets
with open('./data/spam_detection/train_data_bert.pkl', 'wb') as f:
    pickle.dump((train_x, train_y), f)

with open('./data/spam_detection/test_data_bert.pkl', 'wb') as f:
    pickle.dump((test_x, test_y), f)

