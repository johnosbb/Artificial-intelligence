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

class SyslogModel(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN = 10):
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x




# Load the pre-trained model
loaded_model = SyslogModel(NUM_FEATURES=8939, NUM_CLASSES=2)
loaded_model.load_state_dict(torch.load('../models/syslog_model_8939.pth'))




# Assuming you have a DataFrame with new data in a column named 'Detail'
# Replace 'your_new_data.csv' with the actual file or DataFrame containing your new data
new_data = pd.read_csv('../data/syslog.csv').dropna()

# Extract the 'Detail' column from the new data
X_unseen = new_data['Detail'].values
X_unseen_sentiment = new_data['Label'].values
#one_hot = CountVectorizer() # convert a collection of text documents to a matrix of token counts.  Each row in the matrix corresponds to a document, and each column corresponds to a unique word (token) in the entire collection of documents.
one_hot = joblib.load('../models/count_vectorizer.joblib')
# Use the same CountVectorizer instance as during training
# If you saved the original CountVectorizer as 'one_hot', use it here
X_unseen_onehot = one_hot.transform(X_unseen)

# Convert the sparse matrix to a dense PyTorch tensor
X_unseen_tensor = torch.Tensor(X_unseen_onehot.toarray())

# Now, X_unseen_tensor is ready to be used with the loaded model for predictions

# Set the model to evaluation mode
loaded_model.eval()

# Make predictions on the new data
with torch.no_grad():
    y_unseen_logits = loaded_model(X_unseen_tensor)
    y_unseen_pred = torch.argmax(y_unseen_logits, dim=1)

# Convert predictions to numpy array
y_unseen_pred_np = y_unseen_pred.squeeze().cpu().numpy()
# Assuming y_unseen_pred_np is your NumPy array
index = 0
for index, (value, sentiment,text) in enumerate(zip(y_unseen_pred_np, X_unseen_sentiment, X_unseen)):  
    if(value == 1) :
        print(f"Index = {index}, Prediction: {value}, Sentiment: {sentiment}, Text: {text}")