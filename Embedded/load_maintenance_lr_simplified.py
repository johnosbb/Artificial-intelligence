import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

# Load the dataset
file_path = './data/predictive_maintenance_dataset.csv'
data = pd.read_csv(file_path)

# Split the data into features and target
X = data.drop('Motor Fails', axis=1)
y = data['Motor Fails']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(x_test)

# Train the Logistic Regression model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_balanced_scaled, y_train_balanced)

# Evaluate the model
y_test_pred = model.predict(X_test_scaled)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# Predict probabilities
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_test_prob)

# Avoid division by zero by adding a small value to the denominator
epsilon = 1e-10
f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)

# Find the best threshold based on the maximum F1-score
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]

print(f"Best Threshold: {best_threshold}")

# Apply the best threshold
y_test_pred_adjusted = (y_test_prob > best_threshold).astype(int)

# Evaluate the model with the adjusted threshold
from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_adjusted))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_adjusted))
