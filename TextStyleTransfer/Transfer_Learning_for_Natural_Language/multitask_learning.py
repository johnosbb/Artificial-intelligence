import random
import pandas as pd
import re
import nltk
import time
import sent2vec
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Layer
import email
import tensorflow  as tf

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
                            

IMDB_TASK=True
SPAM_EMAIL_TASK=True
SETUP_SENT2VEVMODEL=True

Nsamp = 1000 # number of samples to generate in each class - 'spam', 'not spam'
maxtokens = 200 # the maximum number of tokens per document
maxtokenlen = 100 # the maximum length of each token


def analyze_object(obj, name = ""):
    print("------------------------Object Analysis Begins--------------------------")
    # Check if it's a Series
    print(f"Object name: {name}")
    if isinstance(obj, pd.Series):
        print("Object is a pandas Series")
        print("Shape:", obj.shape)
        print("Length of Series:", obj.shape[0])
        print("Type of elements:", type(obj.iloc[0]))
        print("First 3 elements:")
        print(obj.head(3))
    # Check if it's a DataFrame
    elif isinstance(obj, pd.DataFrame):
        print("Object is a pandas DataFrame")
        print("Shape:", obj.shape)
        print("Length of Series:", len(obj))
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
        print("Shape:", obj.shape)
        print("Dimensions in this array: ", obj.ndim)
        print("Array Size: ", obj.size)
        print("Length of NumPy Array:", len(obj))
        print("Type of elements:", obj.dtype)
        print("First 3 elements:")
        print(obj[:3])
    # Check if it's a TensorFlow tensor
    elif tf.is_tensor(obj):
        print("Object is a TensorFlow tensor")
        print("Shape:", obj.shape)
        print("Dimensions in this tensor: ", obj.ndim)
        print("Length of TensorFlow Object:", obj.shape[0])
        print("Data type:", obj.dtype)
        print("First 3 elements:")
        print(obj[:3])
    elif isinstance(obj, torch.Tensor):
        print("Object is a PyTorch tensor")
        print("Shape:", obj.shape)
        print("Dimensions in this tensor: ", obj.dim())
        print("Length of PyTorch tensor:", obj.size(0))
        print("Data type:", obj.dtype)
        print("First 3 elements:")
        print(obj[:3])
    else:
        print("Unknown type")

    print("------------------------Object Analysis Ends--------------------------")

def tokenize(row):
    if row is None or row == '':
        tokens = ""
    else:
        tokens = str(row).split(" ")[:maxtokens]
    return tokens

# print(stopwords) # see default stopwords
# it may be beneficial to drop negation words from the removal list, as they can change the positive/negative meaning
# of a sentence
# stopwords.remove("no")
# stopwords.remove("nor")
# stopwords.remove("not")

def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower() # make all characters lower case
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


def assemble_embedding_vectors(data):
    out = None
    for item in data:
        vec = s2v_model.embed_sentence(" ".join(item))
        if vec is not None:
            if out is not None:
                out = np.concatenate((out,vec),axis=0)
            else:
                out = vec                                            
        else:
            pass
        
        
    return out


# shuffle raw data first
def unison_shuffle_data(data, header):
    p = np.random.permutation(len(header))
    data = data[p]
    header = np.asarray(header)[p]
    return data, header

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
            # Determine the maximum shape among all elements
    max_length = 0
    for element in data:
        if isinstance(element, list):
            length = len(element)
            if length > max_length:
                max_length = length
        else:
            print("Element is not a list:", element)
    print(f"Max length is {max_length}") 
    valid_elements = []
    # Iterate over each element in data and check if it can be converted to a NumPy array
    for i, element in enumerate(data):
        try:
            # print(f"Element Length {i} : {len(element)}")
            np_array = np.array(element)
            valid_elements.append(np_array)
        except Exception as e:
            print(f"Error converting element {i} to NumPy array:", e)
    # Pad shorter lists with an empty string ''
    padded_data = [element + [''] * (max_length - len(element)) for element in data]

    data_np = np.array(padded_data)
    data, sentiments = unison_shuffle_data(data_np, sentiments)
    return data, sentiments


if SETUP_SENT2VEVMODEL:
    s2v_model = sent2vec.Sent2vecModel()
    start=time.time()
    s2v_model.load_model('./data/ShallowTransferLearning/sent2vec/wiki_unigrams.bin')
    end = time.time()
    print("Loading the sent2vec embedding took %d seconds"%(end-start))

if IMDB_TASK:
    train_path = os.path.join('./data/ShallowTransferLearning/aclImdb', 'train')
    test_path = os.path.join('./data/ShallowTransferLearning/aclImdb', 'test')
    raw_data, raw_header = load_data(train_path)
    print(raw_data.shape)
    print(len(raw_header))

    # Subsample required number of samples
    random_indices = np.random.choice(range(len(raw_header)),size=(Nsamp*2,),replace=False)
    data_train = raw_data[random_indices]
    header = raw_header[random_indices]

    del raw_data, raw_header # huge and no longer needed, get rid of it

    print("DEBUG::data_train::")
    print(data_train)

    unique_elements, counts_elements = np.unique(header, return_counts=True)
    print("Sentiments and their frequencies:")
    print(f"number of unique elements: {unique_elements}")
    print(f"count of these elements: {counts_elements}")


    EmbeddingVectors = assemble_embedding_vectors(data_train)
    print(EmbeddingVectors)

    data = EmbeddingVectors
    del EmbeddingVectors

    idx = int(0.7*data.shape[0])

    # 70% of data for training
    train_x = data[:idx,:]
    train_y = header[:idx]
    # # remaining 30% for testing
    test_x = data[idx:,:]
    test_y = header[idx:] 

    print("train_x/train_y list details, to make sure it is of the right form:")
    print(len(train_x))
    print(train_x)
    print(train_y[:5])
    print(len(train_y))




    input_shape = (len(train_x[0]),)
    sent2vec_vectors = Input(shape=input_shape)
    dense = Dense(512, activation='relu')(sent2vec_vectors)
    dense = Dropout(0.3)(dense)
    output = Dense(1, activation='sigmoid')(dense) # single binary classifier—is review “positive” or “negative”?
    model = Model(inputs=sent2vec_vectors, outputs=output)

    model.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32,
                        epochs=10, shuffle=True)


if SPAM_EMAIL_TASK:
    # Input data files are available in the "../input/" directory.
    filepath = "./data/spam_detection/emails.csv"

    # Read the enron data into a pandas.DataFrame called emails
    emails = pd.read_csv(filepath)

    print("Successfully loaded {} rows and {} columns!".format(emails.shape[0], emails.shape[1]))
    print(emails.head())

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

    bodies = extract_messages(emails)

    del emails


    bodies_df = pd.DataFrame(random.sample(bodies, 10000))

    del bodies # these are huge, no longer needed, get rid of them

    # expand default pandas display options to make emails more clearly visible when printed
    pd.set_option('display.max_colwidth', 300)

    bodies_df.head() # you could do print(bodies_df.head()), but Jupyter displays this nicer for pandas DataFrames

    filepath = "./data/spam_detection/fradulent_emails.txt"
    with open(filepath, 'r',encoding="latin1") as file:
        data = file.read()

        fraud_emails = data.split("From r")
    # Split on the code word `From r` appearing close to the beginning of each email
      
    del data

    print("Successfully loaded {} spam emails!".format(len(fraud_emails)))

    fraud_bodies = extract_messages(pd.DataFrame(fraud_emails,columns=["message"]))

    del fraud_emails

    fraud_bodies_df = pd.DataFrame(fraud_bodies[1:])

    del fraud_bodies

    fraud_bodies_df.head() # you could do print(fraud_bodies_df.head()), but Jupyter displays this nicer for pandas DataFrames

    # Convert everything to lower-case, truncate to maxtokens and truncate each token to maxtokenlen
    EnronEmails = bodies_df.iloc[:,0].apply(tokenize)
    EnronEmails = EnronEmails.apply(stop_word_removal)
    EnronEmails = EnronEmails.apply(reg_expressions)
    EnronEmails = EnronEmails.sample(Nsamp)

    del bodies_df

    SpamEmails = fraud_bodies_df.iloc[:,0].apply(tokenize)
    SpamEmails = SpamEmails.apply(stop_word_removal)
    SpamEmails = SpamEmails.apply(reg_expressions)
    SpamEmails = SpamEmails.sample(Nsamp)

    del fraud_bodies_df

    raw_data = pd.concat([SpamEmails,EnronEmails], axis=0).values


    print("Shape of combined data is:")
    print(raw_data.shape)
    print("Data is:")
    print(raw_data)

    # create corresponding labels
    Categories = ['spam','notspam']
    header = ([1]*Nsamp)
    header.extend(([0]*Nsamp))

    EmbeddingVectors = assemble_embedding_vectors(raw_data)
    print(EmbeddingVectors)


    data, header = unison_shuffle_data(EmbeddingVectors, header)

    idx = int(0.7*data.shape[0])

    # 70% of data for training
    train_x2 = data[:idx,:]
    train_y2 = header[:idx]
    # # remaining 30% for testing
    test_x2 = data[idx:,:]
    test_y2 = header[idx:] 

    print("train_x2/train_y2 (emails) list details, to make sure it is of the right form:")
    print(len(train_x2))
    print(train_x2)
    print(train_y2[:5])
    print(len(train_y2))

    input_shape = (len(train_x2[0]),)
    sent2vec_vectors = Input(shape=input_shape)
    dense = Dense(512, activation='relu')(sent2vec_vectors)
    dense = Dropout(0.3)(dense)
    output = Dense(1, activation='sigmoid')(dense) #he output indicates a single binary  classifier—is email “spam” or “not spam”?
    model = Model(inputs=sent2vec_vectors, outputs=output)

    model.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_x2, train_y2, validation_data=(test_x2, test_y2), batch_size=32,
                        epochs=10, shuffle=True)


# TensorFlow version: 2.16.1

input1_shape = (len(train_x[0]),)
input2_shape = (len(train_x2[0]),)
sent2vec_vectors1 = Input(shape=input1_shape, name="vector1")
sent2vec_vectors2 = Input(shape=input2_shape, name="vector2")

class ConcatenateLayer(Layer):
    def call(self, inputs, axis=0):
        return tf.concat(inputs, axis=axis)
    


combined = ConcatenateLayer()([sent2vec_vectors1,sent2vec_vectors2],axis=0)
dense1 = Dense(512, activation='relu')(combined)
#dense1 = Dense(512, activation='relu')(sent2vec_vectors1)
dense1 = Dropout(0.3)(dense1)
output1 = Dense(1, activation='sigmoid',name='classification1')(dense1)
output2 = Dense(1, activation='sigmoid',name='classification2')(dense1)

print("Input shapes:")
print("sent2vec_vectors1:", sent2vec_vectors1.shape)
print("sent2vec_vectors2:", sent2vec_vectors2.shape)

print("Output shapes:")
print("output1:", output1.shape)
print("output2:", output2.shape)


model = Model(inputs=[sent2vec_vectors1,sent2vec_vectors2], outputs=[output1,output2])

model.compile(loss={'classification1': 'binary_crossentropy', 
                    'classification2': 'binary_crossentropy'},
              optimizer='adam', metrics=['accuracy', 'accuracy'])


analyze_object(train_x,"train_x")
analyze_object(train_x2,"train_x2")
analyze_object(train_y,"train_y")
analyze_object(train_y2,"train_y2")
analyze_object(test_x,"test_x")
analyze_object(test_x2,"test_x2")
analyze_object(test_y,"test_y")
analyze_object(test_y2,"test_y2")  
                          
history = model.fit([train_x, train_x2], [train_y, train_y2],
                    validation_data=([test_x, test_x2], [test_y, test_y2]),
                     epochs=10, shuffle=True
                    )

# history = model.fit([train_x.reshape(3281, 256, -1), train_x2.reshape(3281, 256, -1)],
#                     [train_y, train_y2],
#                     batch_size=256,  # Adjusted batch size for train_x2
#                     epochs=10, shuffle=True)

# history = model.fit(train_x,train_y,batch_size=256,  # Adjusted batch size for train_x2
# epochs=10, shuffle=True)