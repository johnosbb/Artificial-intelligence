import time
from gensim.models import FastText, KeyedVectors
import re
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Nsamp = 1000 # number of samples to generate in each class - 'spam', 'not spam'
maxtokens = 200 # the maximum number of tokens per document
maxtokenlen = 100 # the maximum length of each token



LOAD_WIKI_VECTORS=True
DOWNLOAD_STOPWORDS=True

if LOAD_WIKI_VECTORS:
    start=time.time()
    FastText_embedding = KeyedVectors.load_word2vec_format("./data/ShallowTransferLearning/wiki.en.vec")
    end = time.time()
    print("Loading the embedding took %d seconds"%(end-start))

def tokenize(row):
    if row is None or row == '':
        tokens = ""
    else:
        tokens = row.split(" ")[:maxtokens]
    return tokens



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


if DOWNLOAD_STOPWORDS:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')    

# print(stopwords) # see default stopwords
# it may be beneficial to drop negation words from the removal list, as they can change the positive/negative meaning
# of a sentence
# stopwords.remove("no")
# stopwords.remove("nor")
# stopwords.remove("not")

def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]
    token = filter(None, token)
    return token

def handle_out_of_vocab(embedding,in_txt):
    out = None
    for word in in_txt:
        try:
            tmp = embedding[word]
            tmp = tmp.reshape(1,len(tmp))
            
            if out is None:
                out = tmp
            else:
                out = np.concatenate((out,tmp),axis=0)
        except:
            pass
    
    return out
        

def assemble_embedding_vectors(data):
    out = None
    for item in data:
        tmp = handle_out_of_vocab(FastText_embedding,item)
        if tmp is not None:
            dim = tmp.shape[1]
            if out is not None:
                vec = np.mean(tmp,axis=0)
                vec = vec.reshape((1,dim))
                out = np.concatenate((out,vec),axis=0)
            else:
                out = np.mean(tmp,axis=0).reshape((1,dim))                                            
        else:
            pass
        
        
    return out


import os
import numpy as np

# shuffle raw data first
def unison_shuffle_data(data, header):
    p = np.random.permutation(len(header))
    data = data[p]
    header = np.asarray(header)[p]
    return data, header


# load data in appropriate form
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
    #max_shape = max([element.shape for element in data], key=len)
    # Initialize an empty array with the maximum shape
    # data_np = np.zeros((len(data),) + (max_length,))
    # for i, element in enumerate(data):
    #     data_np[i, :len(element)] = element

# Initialize an empty list to store valid elements
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

train_path = os.path.join('./data/ShallowTransferLearning/aclImdb', 'train')
test_path = os.path.join('./data/ShallowTransferLearning/aclImdb', 'test')
raw_data, raw_header = load_data(train_path)

print(raw_data.shape)
print(len(raw_header))
# Subsample required number of samples
random_indices = np.random.choice(range(len(raw_header)),size=(Nsamp*2,),replace=False)
data_train = raw_data[random_indices]
header = raw_header[random_indices]

print("DEBUG::data_train::")
print(data_train)

unique_elements, counts_elements = np.unique(header, return_counts=True)
print("Sentiments and their frequencies:")
print(f"number of unique elements: {unique_elements}")
print(f"count of these elements: {counts_elements}")


EmbeddingVectors = assemble_embedding_vectors(data_train)
print("Embedding Vectors:\n")
print(EmbeddingVectors)

data = EmbeddingVectors

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


def fit(train_x,train_y):
    model = LogisticRegression()

    try:
        model.fit(train_x, train_y)
    except:
        pass
    return model

model = fit(train_x,train_y)

predicted_labels = model.predict(test_x)
print("DEBUG::The logistic regression predicted labels are::")
print(predicted_labels)



acc_score = accuracy_score(test_y, predicted_labels)

print("The logistic regression accuracy score is::")
print(acc_score)