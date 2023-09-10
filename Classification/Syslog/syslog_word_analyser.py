import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

model_filename = './data/Syslog/random_forest_model_tfid.joblib'
vectorizer_filename = './data/vectorizer_tfid.joblib'

# Load the model from the file
loaded_model = joblib.load(model_filename)

# Access the vectorizer from the loaded_model
vectorizer = joblib.load(vectorizer_filename)

# Specify the word you want to get the probability for
target_word = "error"  # Replace with the word you're interested in

# Find the index of the target word in the TF-IDF vocabulary
word_index = vectorizer.vocabulary_.get(target_word)

# Create a custom feature vector with all zeros except for the target word's position
custom_feature_vector = [0] * len(vectorizer.vocabulary_)
if word_index is not None:
    custom_feature_vector[word_index] = 1

# Convert the custom feature vector into a NumPy array
custom_feature_vector = [custom_feature_vector]

# Use predict_proba with the custom feature vector
probability_for_word = loaded_model.predict_proba(custom_feature_vector)

# Print the probability scores for class 0 and class 1
print(
    f"Probability for '{target_word}' in class 0: {probability_for_word[0][0]:.2f}")
print(
    f"Probability for '{target_word}' in class 1: {probability_for_word[0][1]:.2f}")
