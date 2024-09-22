import numpy as np
import tensorflow as tf

# Load your saved model (choose the format you prefer, .keras or SavedModel)
# Option 1: Load Keras model
model = tf.keras.models.load_model("./models/preventive_forecast.keras")

# Option 2: Load from the SavedModel format
# model = tf.keras.models.load_model("./models/preventive_forecast_saved_model")

# Define your input features (hardcode a single sample)
# Example: Replace these with actual feature values from your dataset
single_sample = np.array([[1700, 25.5, 0.135, 3.75]])  # Hardcoded input features

# Standardize or normalize the single input sample (ensure this matches the training preprocessing)
# Replace 'mean_values' and 'std_values' with the actual mean and standard deviation used in training
mean_values = np.array([1603.866, 24.354, 0.120, 3.494], dtype=np.float32)
std_values = np.array([195.843, 4.987, 0.020, 0.308], dtype=np.float32)

# Scale the input
single_sample_scaled = (single_sample - mean_values) / std_values

# Make a prediction using the single sample
prediction = model.predict(single_sample_scaled)

# Display the result
failure_prob = prediction[0][0]  # Assuming the model outputs a single value representing failure probability
print(f"Motor Failure Prediction Probability: {failure_prob:.6f}")
if failure_prob > 0.5:
    print("Motor Failure Detected")
else:
    print("No Motor Failure Detected")
