import numpy as np
import pandas as pd

# Set the seed for reproducibility
np.random.seed(42)

# Number of data points
n = 1000

# Generate random data for features with specific ranges
rpm = np.random.normal(1600, 200, n)  # RPM centered around 1600 with std dev of 200
temperature = np.random.normal(24, 5, n)  # Temperature centered around 24°C with std dev of 5
vibration = np.random.normal(0.12, 0.02, n)  # Vibration centered around 0.12g with std dev of 0.02
current = np.random.normal(7.5, 0.3, n)  # Current centered around 7.5A with std dev of 0.3

# Initialize the failure indicator
motor_fails = np.zeros(n)

# Define thresholds for failure conditions
high_temp_threshold = 30
low_rpm_threshold = 1500
high_vibration_threshold = 0.60
abnormal_current_low_threshold = 4.2
abnormal_current_high_threshold = 10.8

# Determine failure based on defined thresholds
for i in range(n):
    if (temperature[i] > high_temp_threshold or
        rpm[i] < low_rpm_threshold or
        vibration[i] > high_vibration_threshold or
        current[i] < abnormal_current_low_threshold or
        current[i] > abnormal_current_high_threshold):
        motor_fails[i] = np.random.choice([1, 0], p=[0.8, 0.2])  # Higher chance of failure if conditions met
    else:
        motor_fails[i] = np.random.choice([0, 1], p=[0.98, 0.02])  # Lower chance of failure otherwise

# Create a DataFrame
data = pd.DataFrame({
    'RPM': rpm,
    'Temperature (°C)': temperature,
    'Vibration (g)': vibration,
    'Current (A)': current,
    'Motor Fails': motor_fails
})

# Display the first few rows of the dataset
print(data.head())

# Save the data to a CSV file
file_path = './data/predictive_maintenance_dataset.csv'
data.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")


# NOTE: Data quality and the strength of feature-target relationships are crucial for model performance.
# This dataset was designed with clear, strong relationships between features and the target.
#  This helps the model to identify patterns and make accurate predictions.
