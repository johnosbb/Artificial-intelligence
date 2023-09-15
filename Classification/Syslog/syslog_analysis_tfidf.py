import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model_tfid.joblib'
vectorizer_filename = './data/vectorizer_tfid.joblib'

# Preprocess log lines (remove timestamps and other noise)

df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
labels = df["Label"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()  # Use TfidfVectorizer instead of CountVectorizer

# Convert preprocessed log lines to TF-IDF feature vectors
feature_vectors = vectorizer.fit_transform(details)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, labels, test_size=0.2, random_state=42)


# n_estimators: The number of trees in the forest. Increasing the number of trees can improve model performance up to a point. However, more trees also mean longer training times.
# max_depth: The maximum depth of each tree in the forest. Increasing max_depth can make the trees more complex and potentially capture more intricate patterns in the data. Be cautious not to set it too high to avoid overfitting.
# min_samples_split: The minimum number of samples required to split an internal node. Increasing this parameter can make the tree less likely to split, which can reduce overfitting.
# min_samples_leaf: The minimum number of samples required to be at a leaf node. Similar to min_samples_split, increasing this parameter can regularize the tree.
# max_features: The number of features to consider when looking for the best split. You can experiment with different values, such as  (sqrt(n_features)), 'log2' (log2(n_features)), or an integer representing the number of features.
# class_weight: If your dataset is imbalanced (which is common in anomaly detection tasks like log analysis), you can set class_weight to 'balanced' to automatically adjust the weights of classes inversely proportional to their frequencies.
# bootstrap: Whether or not to use bootstrap samples when building trees. Setting it to False can be useful if you want to disable bootstrapping.
# random_state: Set a specific random seed for reproducibility.
# Train a machine learning model (Random Forest classifier in this example)
# Train a machine learning model (Random Forest classifier) with modified hyperparameters
clf = RandomForestClassifier(
    # n_estimators=100,  # Increase the number of trees
    # max_depth=None,    # Allow trees to grow until fully developed
    # min_samples_split=2,  # Reduce the minimum samples required to split
    # min_samples_leaf=1,   # Allow smaller leaf nodes
    # max_features='sqrt',  # Consider all features for splitting
    class_weight='balanced',  # Adjust class weights for imbalanced data
    # bootstrap=True,   # Use bootstrapped samples
    random_state=42   # Set a specific random seed for reproducibility
)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(clf, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
