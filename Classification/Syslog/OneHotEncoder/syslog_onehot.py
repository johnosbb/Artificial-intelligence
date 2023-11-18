import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

# Reference: https://jeancochrane.com/blog/pytorch-functional-api

syslog_file = '../data/syslog.csv'
df = pd.read_csv(syslog_file).dropna()

BATCH_SIZE = 512 # smaller batch size produces lower generalization errors. Is is easier to avoid constraints with memory or GPU.
NUM_EPOCHS = 100

X = df['Detail'].values
y = df['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=123)
print(f"X train: {X_train.shape}, y train: {y_train.shape}\nX test: {X_test.shape}, y test: {y_test.shape}")

one_hot = CountVectorizer() # convert a collection of text documents to a matrix of token counts.  Each row in the matrix corresponds to a document, and each column corresponds to a unique word (token) in the entire collection of documents.
X_train_onehot = one_hot.fit_transform(X_train) #  learn the parameters from the training data and transform that data.
X_test_onehot = one_hot.transform(X_test) # just transform the test data

number_of_features = X_train_onehot.shape[1]
print(f"X_train_onehot.shape[1] = {number_of_features}")

# Save the CountVectorizer from the training stage
joblib.dump(one_hot, '../models/count_vectorizer.joblib')


class SyslogData(Dataset): # inherits the Dataset Class
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.Tensor(X.toarray())
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.len = len(self.X)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

train_ds = SyslogData(X= X_train_onehot, y = y_train)
test_ds = SyslogData(X_test_onehot, y_test)

train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=27932) # 27932 is a factor of the total size of the data. In this case there are (27932 * 2 = 55864) samples and we use half for test

class SyslogModel(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN = 10, loss_function=None, optimizer=None):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, HIDDEN) # nn.Linear is a linear transformation layer, also known as a fully connected or dense layer. NUM_FEATURES represents the number of input features (neurons) to this layer. HIDDEN represents the number of neurons in the hidden layer.
        self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU() # nn.ReLU is the rectified linear unit activation function. It introduces non-linearity to the model by outputting the input directly if it is positive; otherwise, it will output zero.
        self.log_softmax = nn.LogSoftmax(dim=1) # nn.LogSoftmax is used to convert the raw output scores from the output layer into log probabilities.
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

model = SyslogModel(NUM_FEATURES = number_of_features, NUM_CLASSES = 2,loss_function=nn.CrossEntropyLoss,optimizer=torch.optim.AdamW)
model.initialize_optimizer()
model.initialize_loss()

# # nn.CrossEntropyLoss is a PyTorch loss function specifically designed for multiclass classification problems.
# # It combines the softmax activation function and the negative log-likelihood loss.
# # It expects raw logits (unnormalized scores) as input and internally applies the softmax function to convert
# # them into probabilities before computing the loss.
# criterion = nn.CrossEntropyLoss()
# # torch.optim.AdamW is an implementation of the Adam optimizer with weight decay.
# # AdamW is a variant of the Adam optimizer that includes L2 weight decay (weight regularization).
# optimizer = torch.optim.AdamW(model.parameters())


train_losses = []
for e in range(NUM_EPOCHS):
    curr_loss = 0
    for X_batch, y_batch in train_loader: # this will allow us to batch the data on each Epoch
        # In PyTorch, optimizers know nothing about the training loop, 
        # so they will continue to accumulate gradients indefinitely unless instructed to stop.
        # Hence, to initiate a new forward and backward pass,
        # we need to start by clearing out any gradients that may have previously been calculated:
        model.zero_gradient()
        y_pred_log = model.make_prediction(X_batch) # this invokes the forward pass in the model
        loss = model.loss(y_pred_log, y_batch.long()) # calculate our losses        
        curr_loss += loss.item()
        # With loss.backward() PyTorch is obscuring the fact that it is using the layers contained by model 
        # to propagate the gradient backward through the network -- 
        # layers which have been passed through to output by the loss function,
        # in order to calculate the gradient in loss.backward().
        # Instead of making this shared state clear, the API obscures it, 
        # returning None and mutating gradient state in-place.
        loss.backward() # triggers automatic gradient calculation
        model.optimizer.step()  # Update the weights
    train_losses.append(curr_loss)
    print(f"Epoch {e}, Loss: {curr_loss}")

loss_plot =  sns.lineplot(x=list(range(len(train_losses))), y= train_losses)
loss_plot.figure.savefig('loss_plot.svg')
#plt.show()


with torch.no_grad(): # This context manager is used to turn off gradient computation within its scope. It means any operations inside this block won't track gradients for subsequent backward passes.
    for X_batch, y_batch in test_loader:
        y_test_pred_log = model.make_prediction(X_batch) # The model is used to make predictions 
        y_test_pred = torch.argmax(y_test_pred_log, dim = 1) #  This line finds the indices of the maximum values along the specified dimension (dim=1), effectively giving the predicted class labels. The result, y_test_pred, contains the predicted labels for the batch.

y_test_pred_np = y_test_pred.squeeze().cpu().numpy() # This removes dimensions of size 1 from the shape of y_test_pred. It is commonly used to remove unnecessary singleton dimensions.
print(f"y_test_pred_np.shape {y_test_pred_np.shape}  y_test_pred {y_test_pred.shape}")
print(f"y_test.shape {y_test.shape}")
acc = accuracy_score(y_pred=y_test_pred_np, y_true = y_test)
f"The accuracy of the model is {np.round(acc, 3)*100}%."
most_common_cnt = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %")
heatmap_plot = sns.heatmap(confusion_matrix(y_test_pred_np, y_test), annot=True, fmt=".0f")
#plt.show()
heatmap_plot.figure.savefig('heatmap.svg')
#sns.heatmap_plot.figure.savefig('heatmap.svg')
# Save the trained model
torch.save(model.state_dict(), f"../models/syslog_model_{number_of_features}.pth") # if we save the complete model we are dependent on directory structures. We just save the state dictionary. This has learnable parameters and registered buffers 
