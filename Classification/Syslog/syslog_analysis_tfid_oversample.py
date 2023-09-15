import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Import RandomOverSampler from imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
import joblib

filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model_tfid_with_oversampling.joblib'
vectorizer_filename = './data/vectorizer_tfid_with_oversampling.joblib'

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

# Random oversampling
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_oversampled, y_train_oversampled = oversampler.fit_resample(
    X_train, y_train)

# Train a machine learning model (Random Forest classifier) with modified hyperparameters
# clf = RandomForestClassifier(
#     class_weight='balanced_subsample',
#     max_depth=10,
#     max_features='sqrt',
#     min_samples_leaf=1,
#     min_samples_split=2,
#     n_estimators=50,
#     random_state=42
# )

clf = RandomForestClassifier(
    # n_estimators=100,  # Increase the number of trees
    max_depth=10,    # Allow trees to grow until fully developed
    # min_samples_split=2,  # Reduce the minimum samples required to split
    # min_samples_leaf=1,   # Allow smaller leaf nodes
    # max_features='sqrt',  # Consider all features for splitting
    class_weight='balanced',  # Adjust class weights for imbalanced data
    # bootstrap=True,   # Use bootstrapped samples
    random_state=42   # Set a specific random seed for reproducibility
)
clf.fit(X_train_oversampled, y_train_oversampled)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(clf, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
