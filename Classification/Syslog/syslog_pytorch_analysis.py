
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Define a simple feedforward neural network model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Your file paths and preprocessing here...
filtered_syslog_file_path = './data/syslog.cvs'
model_filename = './data/pytorch_model.pth'
vectorizer_filename = './data/vectorizer.joblib'

# Preprocess log lines (remove timestamps and other noise)
df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
labels = df["Label"]

# Create a bag-of-words (BoW) vectorizer
vectorizer = CountVectorizer()
feature_vectors = vectorizer.fit_transform(details)

# Convert data to PyTorch tensors
X = torch.Tensor(feature_vectors.toarray())
y = torch.Tensor(labels.values).unsqueeze(1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train.toarray())
y_train_tensor = torch.Tensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.Tensor(X_test.toarray())
y_test_tensor = torch.Tensor(y_test.values).unsqueeze(1)

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define hyperparameters
input_size = X_train.shape[1]
hidden_size = 128
output_size = 1  # Binary classification

# Initialize the model and optimizer
model = SimpleClassifier(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
y_pred_prob = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        y_pred_prob.extend(outputs.numpy())

y_pred = [1 if prob[0] >= 0.5 else 0 for prob in y_pred_prob]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the PyTorch model
torch.save(model.state_dict(), model_filename)
