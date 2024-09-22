import numpy as np
import pandas as pd
import os

# Set the seed for reproducibility
np.random.seed(42)

# Number of data points
n = 1000

# Generate random data for features with specific ranges
rpm = np.random.normal(1600, 200, n)  # RPM centered around 1600 with std dev of 200
temperature = np.random.normal(24, 5, n)  # Temperature centered around 24°C with std dev of 5
vibration = np.random.normal(0.12, 0.02, n)  # Vibration centered around 0.12g with std dev of 0.02
current = np.random.normal(3.5, 0.3, n)  # Current centered around 3.5A with std dev of 0.3

# Initialize the failure indicator
motor_fails = np.zeros(n)

# Define thresholds for failure conditions
high_temp_threshold = 30
low_rpm_threshold = 1500
high_vibration_threshold = 0.60
abnormal_current_low_threshold = 0.2
abnormal_current_high_threshold = 10.8

for i in range(n):
    # Check if failure conditions are met
    condition_met = (temperature[i] > high_temp_threshold or
                     rpm[i] < low_rpm_threshold or
                     vibration[i] > high_vibration_threshold or
                     current[i] < abnormal_current_low_threshold or
                     current[i] > abnormal_current_high_threshold)

    if condition_met:
        motor_fails[i] = np.random.choice([1, 0], p=[0.8, 0.2])  # Higher chance of failure if conditions met
    else:
        motor_fails[i] = np.random.choice([0, 1], p=[0.98, 0.02])  # Lower chance of failure otherwise

    # Print warning if a failure is set but no threshold conditions are met
    if motor_fails[i] == 1 and not condition_met:
        print(f"Warning: Failure recorded at index {i} but no threshold conditions met.")
        print(f"Values - RPM: {rpm[i]}, Temperature: {temperature[i]}, Vibration: {vibration[i]}, Current: {current[i]}\n")

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

# We calculate the scales and zero-points purely for reference in any subsequent trouble shooting we may need to do
# Calculate input scales and zero-points for normalization
min_values = data[['RPM', 'Temperature (°C)', 'Vibration (g)', 'Current (A)']].min().values
max_values = data[['RPM', 'Temperature (°C)', 'Vibration (g)', 'Current (A)']].max().values
input_scale = (max_values - min_values) / 255.0
input_zero_point = min_values

# Calculate output scale and zero-point
min_output = 0  # Assuming binary classification, minimum output is 0
max_output = 1  # Maximum output is 1 for probabilities
output_scale = (max_output - min_output) / 255.0
output_zero_point = min_output

# Display input and output scales and zero points
print("\nInput Scales:", input_scale)
print("Input Zero Points:", input_zero_point)
print("\nOutput Scale:", output_scale)
print("Output Zero Point:", output_zero_point)


# Count the occurrences of each class (0 or 1)
counts = data['Motor Fails'].value_counts()

# Calculate the ratio for each class
ratios = counts / len(data)

print("Class distribution (counts):")
print(counts)

print("\nClass distribution (ratios):")
print(ratios)

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Running this code in the current working directory:", current_directory)


# Save the data to a CSV file
file_path = './data/predictive_maintenance_dataset.csv'
data.to_csv(file_path, index=False)
print(f"Dataset saved to {file_path}")


# NOTE: Data quality and the strength of feature-target relationships are crucial for model performance.
# This dataset was designed with clear, strong relationships between features and the target.
#  This helps the model to identify patterns and make accurate predictions.
