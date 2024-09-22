import pandas as pd

# Load the dataset
file_path = './data/predictive_maintenance_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Extract input features (assuming the columns are as described)
input_features = data[['RPM', 'Temperature (°C)', 'Vibration (g)', 'Current (A)']]

# Save the input features to dataset.txt
with open('./data/dataset.txt', 'w') as f:
    for index, row in input_features.iterrows():
        # Convert each row to a comma-separated string
        line = ','.join(map(str, row))
        f.write(line + '\n')

print('Dataset file dataset.txt created successfully.')
