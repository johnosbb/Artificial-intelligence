

import numpy as np
import pandas as pd
import email
import re
import nltk
import random
 
nltk.download('stopwords')
from nltk.corpus import stopwords
 

Nsamp = 1000
maxtokens = 50
maxtokenlen = 20
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






def tokenize(row):
    if row in [None,'']:
        tokens = ""
    else:
        tokens = str(row).split(" ")[:maxtokens]
    return tokens

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

print("Shape of combined data represented as NumPy array is:")
print(raw_data.shape)
print("Data represented as NumPy array is:")
print(raw_data)

Categories = ['spam','notspam']
header = ([1]*Nsamp)
header.extend(([0]*Nsamp))


# Add categories and header to the DataFrame
raw_data_df['Category'] = header
raw_data_df.columns = Categories + ['Category']

# Save the DataFrame to CSV
raw_data_df.to_csv('./data/spam_detection/rawdata.csv', index=False)

EnronSpamBag = assemble_bag(raw_data)
# The column labels indicate words in the vocabulary of the bag-of-words model,
# and the numerical entries in each row correspond to the frequency counts of 
# each such word for each of the 2,000 emails in our dataset.
# This is an extremely sparse DataFrame—it consists mostly of values of 0.
print(f"EnronSpamBag = {EnronSpamBag}")
# This next line creates a list called predictors containing the column names of the DataFrame EnronSpamBag.
# In the context of machine learning or statistical modeling, these columns are often referred to as "predictors" or "features."
# Each predictor corresponds to a unique token in the bag-of-words representation, 
# and the goal may be to use these predictors to predict or classify some outcome variable.
predictors = [column for column in EnronSpamBag.columns] 

# we split it into independent training and testing, or validation, sets. 
# This will allow us to evaluate the performance of the classifier on a set
# of data that was not used for training—an important thing to ensure in machine learning practice.
# We elect to use 70% of the data for training and 30% for testing/validation afterward.
data, header = unison_shuffle_data(EnronSpamBag.values, header)
idx = int(0.7*data.shape[0])
train_x = data[:idx]
train_y = header[:idx]
test_x = data[idx:]
test_y = header[idx:]