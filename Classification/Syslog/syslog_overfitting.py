import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Load your data and preprocess it as you did before
# ...
filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model.joblib'
vectorizer_filename = './data/vectorizer.joblib'  # Choose a filename


# Preprocess log lines (remove timestamps and other noise)

df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
# Labels indicating whether a particular event of interest occurred (1 for occurrence, 0 for non-occurrence)
labels = df["Label"]

# Create a bag-of-words (BoW) vectorizer
vectorizer = CountVectorizer()

# Convert preprocessed log lines to feature vectors
feature_vectors = vectorizer.fit_transform(details)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, labels, test_size=0.2, random_state=42)

# Define a range of values for the hyperparameter you want to tune (e.g., max_depth)
param_range = np.arange(1, 21)  # Adjust the range as needed

# Calculate training and validation scores for different hyperparameter values
train_scores, test_scores = validation_curve(
    # Your Random Forest classifier
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    X_train, y_train,
    param_name="max_depth",  # Hyperparameter to tune (e.g., max_depth)
    param_range=param_range,
    cv=5,  # 5-fold cross-validation (adjust as needed)
    # Use accuracy as the evaluation metric (adjust as needed)
    scoring="accuracy"
)

# Calculate mean and standard deviation of training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.title("Validation Curve for Random Forest")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.grid()

plt.plot(param_range, train_mean, label="Training Score", marker="o")
plt.fill_between(param_range, train_mean - train_std,
                 train_mean + train_std, alpha=0.2)

plt.plot(param_range, test_mean, label="Test Score", marker="o")
plt.fill_between(param_range, test_mean - test_std,
                 test_mean + test_std, alpha=0.2)

plt.legend(loc="best")
plt.show()
