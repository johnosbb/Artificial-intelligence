
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve,auc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer # is a Python library that provides pre-trained models for generating numerical representations (embeddings) of sentences. These embeddings are designed to capture the semantic meaning of sentences, allowing for various natural language processing (NLP) tasks such as text similarity, clustering, and classification.


syslog_file = '../data/syslog.csv'
df = pd.read_csv(syslog_file).dropna()

BATCH_SIZE = 512 # smaller batch size produces lower generalization errors. Is is easier to avoid constraints with memory or GPU.
NUM_EPOCHS = 100
SYSLOG_PICKLE="../model/syslog_X.pkl"
X = df['Detail'].values
y = df['Label'].values


BATCH_SIZE = 128
NUM_EPOCHS = 100 # Overfitting: Continuing training for too many epochs can lead to overfitting. Overfitting occurs when a model becomes too specialized in the training data and doesn't generalize well to new, unseen data. If the loss on the training set continues to decrease while the loss on a validation set (or new data) starts to increase, it's a sign of overfitting.
MAX_FEATURES = 10
CREATE_ENCODINGS = False

# https://www.sbert.net/
emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # all-mpnet-base-v1': This is specifying the specific pre-trained model to be loaded. In this case, it's using the model named all-mpnet-base-v1 from the sentence-transformers library. The all-mpnet-base-v1 model is based on the MPNet (Multilingual Pre-trained BERT) architecture.

# sentences = [ "Each sentence is converted"]
# embeddings = emb_model.encode(sentences)
# print(embeddings.squeeze().shape)

# The encode method uses the pre-trained model (all-mpnet-base-v1 in this case) to convert each sentence in the provided
# list into a numerical embedding.
# These embeddings are high-dimensional vectors that represent the semantic content of the corresponding sentences.
# The dimensionality of these embeddings is determined by the architecture of the pre-trained model.
# The model considers the entire sentence context when generating embeddings for each word/token in that sentence.
# This is in contrast to simpler word embeddings like Word2Vec or GloVe, 
# which typically generate fixed embeddings for individual words without considering the context in which the words appear.

if (CREATE_ENCODINGS):
    X = emb_model.encode(df['Detail'].values) #This method takes a list of strings (or a single string) and returns the embeddings (numerical representations) for those sentences.
    with open(SYSLOG_PICKLE, "wb") as output_file:
        pickle.dump(X, output_file)
else:
    with open(SYSLOG_PICKLE, "rb") as input_file:
        X = pickle.load(input_file)



# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=123)
print(f"X train: {X_train.shape}, y train: {y_train.shape}\nX test: {X_test.shape}, y test: {y_test.shape}")



class SyslogData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.len = len(self.X)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

train_ds = SyslogData(X= X_train, y = y_train)
test_ds = SyslogData(X_test, y_test)

train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=30000)

number_of_features = X_train.shape[1]
print(f"X_train.shape[1] = {number_of_features}")


class SyslogModel(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN = 10,loss_function=None, optimizer=None):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_function = loss_function
        self.optimizer_function = optimizer
    
    def make_prediction(self,x):
            return self.forward(x) 
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x

    def zero_gradient(self):
        if self.optimizer is not None:
                self.optimizer.zero_grad()
        else:
            raise ValueError("Optimizer not provided during initialization.")
    
    def initialize_loss(self):
        if self.loss_function is not None:
            self.loss =  self.loss_function()
        else:
            raise ValueError("Loss object not provided during initialization.")
        
            
    def initialize_optimizer(self):
        if self.optimizer_function is not None:
            self.optimizer =  self.optimizer_function(self.parameters())
        else:
            raise ValueError("Optimizer not provided during initialization.")



# Assuming y_true and y_score are your PyTorch tensors
# y_true contains true labels (0 or 1)
# y_score contains predicted scores or probabilities
# True Positive Rate: TPR= True Positives/ (False Negatives + True Positives)
# The True Positive Rate measures the proportion of actual positive instances that are correctly predicted as positive.
# It is also known as Sensitivity or Recall.
# False Positive Rate: # FPR = False Positives/ (False Positives + True Negatives)
# The False Positive Rate measures the proportion of actual negative instances that are incorrectly predicted as positive.

def plot_roc(y_true,y_score):
    # Convert PyTorch tensors to NumPy arrays if required
    # y_true_np = y_true.cpu().numpy()
    # y_score_np = y_score.cpu().numpy()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    #plt.show()
    plt.savefig('roc_plot_embeddings.svg', format='svg')
    
    
model = SyslogModel(NUM_FEATURES = number_of_features, NUM_CLASSES = 2,loss_function=nn.CrossEntropyLoss,optimizer=torch.optim.AdamW)
model.initialize_optimizer()
model.initialize_loss()


train_losses = []
for e in range(NUM_EPOCHS):
    curr_loss = 0
    for X_batch, y_batch in train_loader:
        # In PyTorch, optimizers know nothing about the training loop, 
        # so they will continue to accumulate gradients indefinitely unless instructed to stop.
        # Hence, to initiate a new forward and backward pass,
        # we need to start by clearing out any gradients that may have previously been calculated:
        model.zero_gradient()
        y_pred_log = model.make_prediction(X_batch) # this invokes the forward pass in the model
        loss = model.loss(y_pred_log, y_batch.long()) # calculate our losses  
        # Early stopping check     
        curr_loss += loss.item()
        loss.backward() # triggers automatic gradient calculation
        model.optimizer.step()  # Update the weights
    train_losses.append(curr_loss)
    print(f"Epoch {e}, Loss: {curr_loss}")

loss_plot =  sns.lineplot(x=list(range(len(train_losses))), y= train_losses)
loss_plot.figure.savefig('loss_plot_embeddings.svg')



with torch.no_grad(): # This context manager is used to turn off gradient computation within its scope. It means any operations inside this block won't track gradients for subsequent backward passes.
    y_test_pred_total = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_test_pred_log = model.make_prediction(X_batch)
            y_test_pred = torch.argmax(y_test_pred_log, dim = 1)
            y_test_pred_total.extend(y_test_pred.cpu().numpy())
    len(y_test_pred_total)


y_test_pred_np = y_test_pred.squeeze().cpu().numpy()

acc = accuracy_score(y_pred=y_test_pred_np, y_true = y_test)
print(f"The accuracy of the model is {np.round(acc, 3)*100}%.")

most_common_cnt = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %") # This line calculates the percentage of the dataset that corresponds to the most common class and rounds the result to one decimal place. It represents the accuracy of a very simple baseline classifier that always predicts the most common class. This baseline is often referred to as a "majority class classifier" or a "naive classifier."
heatmap_plot = sns.heatmap(confusion_matrix(y_test_pred_np, y_test), annot=True, fmt=".0f")
#plt.show()
heatmap_plot.figure.savefig('heatmap_embeddings.svg')
plot_roc(y_test_pred_np,y_test)