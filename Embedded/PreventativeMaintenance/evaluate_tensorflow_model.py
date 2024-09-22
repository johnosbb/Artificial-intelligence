import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your test data (make sure it is preprocessed)
file_path = './data/predictive_maintenance_dataset.csv'  # Replace with your actual test data file path
test_data = pd.read_csv(file_path)

# Assuming the test data contains the same features as the training data
X_test = test_data.drop('Motor Fails', axis=1)  # Replace 'Motor Fails' with your actual target column name

# Load your saved model (choose the format you prefer, .keras or SavedModel)
# Option 1: Load Keras model
model = tf.keras.models.load_model("./models/preventive_forecast.keras")

# Option 2: Load from the SavedModel format
# model = tf.keras.models.load_model("./models/preventive_forecast_saved_model")

# Standardize or normalize the test data (ensure this matches the training preprocessing)
# Replace 'mean_values' and 'std_values' with the actual mean and standard deviation used in training
mean_values = np.array([1603.866, 24.354, 0.120, 3.494], dtype=np.float32)
std_values = np.array([195.843, 4.987, 0.020, 0.308], dtype=np.float32)

X_test_scaled = (X_test - mean_values) / std_values

# Make predictions using the test set
predictions = model.predict(X_test_scaled)

# Display the results
for i, pred in enumerate(predictions):
    failure_prob = pred[0]  # Assuming the model outputs a single value representing failure probability
    print(f"Motor Failure Prediction Probability for sample {i+1}: {failure_prob:.6f}")
    if failure_prob > 0.5:
        print("Motor Failure Detected")
    else:
        print("No Motor Failure Detected")
