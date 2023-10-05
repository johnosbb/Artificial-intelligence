import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


# Read data
data = pd.read_csv("./data/sonar.csv", header=None)
# The .iloc method is used to select rows and columns from the "data" DataFrame. [:, 0:60] selects all rows (:) and the columns with integer indices from 0 to 59, effectively extracting the first 60 columns.
X = data.iloc[:, 0:60]
# [:, 60] selects all rows and the column with index 60, which corresponds to a specific column in the DataFrame. In this case these will be the labels
y = data.iloc[:, 60]


# The LabelEncoder is to map each label to an integer. In this case, there are only two labels and they will become 0 and 1.
# Using it, you need to first call the fit() function to make it learn what labels are available. Then call transform() to do the actual conversion.
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
# this will print the classes found by the encoder in this ['M' 'R']
print(encoder.classes_)
print(y)

# We convert them into PyTorch tensors as this is the format a PyTorch model would like to work with.
X = torch.tensor(X.values, dtype=torch.float32)
y1 = torch.tensor(y, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
# .reshape(-1, 1): This is the reshape operation.
# The -1 as the first dimension means that PyTorch should automatically calculate the size of that dimension
# based on the size of the original data and the specified column size, which is 1 in this case.
# This operation effectively converts the tensor from a 1D tensor to a 2D tensor with one column.
# Some machine learning libraries and models, especially those in deep learning frameworks like PyTorch and TensorFlow,
# expect target variables to be 2D tensors. This is because many neural network architectures are designed to work with batched data,
# and using 2D tensors simplifies the handling of batched data.


# Concept of a "wider model" in the context of neural networks or deep learning.
# Input Data: The input data in this example has 60 features. These features could represent various characteristics, measurements, or attributes of the data points you're working with. Each feature serves as input information for a machine learning model.
# Binary Variable: The goal of the model is to predict a binary variable. A binary variable typically has only two possible values, such as 0 and 1, or "yes" and "no." In this case, the model is trying to make binary predictions based on the input features.
# Wide Model: When we say a model is "wide," it means that it has a relatively large number of neurons (also called units or nodes) in its layers, particularly the hidden layers. In this example, it suggests creating a neural network model with one hidden layer that contains 180 neurons.
# Neurons in the Hidden Layer: The choice of having 180 neurons in the hidden layer is significant. It is stated as "three times the input features" (60 features * 3 = 180 neurons). This choice of making the hidden layer three times wider than the input features is a design decision. A wider hidden layer can capture more complex patterns and relationships in the data but may also require more data and computational resources.

class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(60, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
