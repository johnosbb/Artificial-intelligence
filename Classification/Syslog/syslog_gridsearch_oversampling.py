import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV
# Import RandomOverSampler from imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
import joblib

filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model_tfid.joblib'
vectorizer_filename = './data/vectorizer_tfid.joblib'

# Preprocess log lines (remove timestamps and other noise)
df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
labels = df["Label"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(details)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, labels, test_size=0.2, random_state=42)

# Random oversampling
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_oversampled, y_train_oversampled = oversampler.fit_resample(
    X_train, y_train)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(
    random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the grid search to your training data
grid_search.fit(X_train_oversampled, y_train_oversampled)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Use the best estimator for predictions
y_pred = best_estimator.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(best_estimator, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
