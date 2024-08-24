import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers


SHOW_PLOTS = False
# Define the CSV filename
csv_filename = './data/aggregated_weather_data.csv'

# Initialize empty lists to store temperature, humidity, and precipitation values
t_list = []
h_list = []
p_list = []
dates = []
rain_day = []


def scaling(val, avg, std):
 return (val - avg) / (std)



# Open the CSV file and read the data
with open(csv_filename, 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        try:
            # Extract and clean temperature, humidity, and precipitation values
            temp = float(row['average_temperature'].replace(' °C', '').strip())
            humidity = float(row['humidity'].replace('%', '').strip())
            precipitation = float(row['precipitation'].replace(' mm', '').strip())
            date = row['date']  # Extract the date
            
            # Append values to the lists
            t_list.append(temp)
            h_list.append(humidity)
            p_list.append(precipitation)
            dates.append(date)

            # Determine if it rained and append to rain_day list
            if precipitation > 0.0:
                rain_day.append(1)  # It rained
            else:
                rain_day.append(0)  # No rain
        except ValueError as e:
            print(f"Error processing row {row}: {e}")

if SHOW_PLOTS:
        # Define the interval for date ticks
        tick_interval = max(1, len(dates) // 10)  # Show at most 10 date labels

        # Create the plot
        plt.figure(figsize=(12, 9))

        # Plot temperature
        plt.subplot(3, 1, 1)
        plt.plot(dates, t_list, label='Temperature (°C)', color='tab:red')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Over Time')
        plt.xticks(ticks=dates[::tick_interval], rotation=45)
        plt.legend()

        # Plot humidity
        plt.subplot(3, 1, 2)
        plt.plot(dates, h_list, label='Humidity (%)', color='tab:blue')
        plt.xlabel('Date')
        plt.ylabel('Humidity (%)')
        plt.title('Humidity Over Time')
        plt.xticks(ticks=dates[::tick_interval], rotation=45)
        plt.legend()

        # Plot precipitation
        plt.subplot(3, 1, 3)
        plt.plot(dates, p_list, label='Precipitation (mm)', color='tab:green')
        plt.xlabel('Date')
        plt.ylabel('Precipitation (mm)')
        plt.title('Precipitation Over Time')
        plt.xticks(ticks=dates[::tick_interval], rotation=45)
        plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Show the plot
        plt.show()


# Create a table with headers
table_data = zip(dates, t_list, h_list, p_list, rain_day)
headers = ["Date", "Temperature (°C)", "Humidity (%)", "Precipitation (mm)", "Rain (1=Yes, 0=No)"]

# Print the table
print(tabulate(table_data, headers=headers, tablefmt="grid"))
csv_header = ["Temp", "Hum", "Rain"]
dataset_df = pd.DataFrame(list(zip(
t_list, h_list, rain_day)), columns = csv_header)


# Step 1: Total number of days
total_days = len(rain_day)
rainy_days = sum(rain_day)

# Step 3: Calculate percentage of days it rained
percentage_rainy_days = (rainy_days / total_days) * 100

print(f"Percentage of days it rained: {percentage_rainy_days:.2f}%")

t_avg = mean(t_list)
h_avg = mean(h_list)
t_std = std(t_list)
h_std = std(h_list)

print("Temperature - [MEAN, STD] ", round(t_avg, 5), round(t_std, 5))
print("Humidity - [MEAN, STD] ", round(h_avg, 5), round(h_std, 5))

# For example, the humidity is between 0 and 100, while the temperature on the Celsius scale can 
# be negative and have a smaller positive numerical range than humidity.
# We use scaling 
# Generally, if the input features have different numerical ranges, the ML model may be influenced 
# more by those with larger values, impacting its effectiveness. Therefore, the input features must 
# be rescaled to ensure that each input feature contributes equally during training.
# Z-score is a common scaling technique adopted in neural networks


dataset_df['Temp'] = dataset_df['Temp'].apply(lambda x: scaling(x, t_avg, t_std))
dataset_df['Hum'] = dataset_df['Hum'].apply(lambda x: scaling(x, h_avg, h_std))

# Extract the input features (x - Temperature and Humidity) and output labels (y - Indication of rain) from the dataset (dataset_df):
f_names = dataset_df.columns.values[0:2]
l_name = dataset_df.columns.values[2]
x = dataset_df[f_names]
y = dataset_df[l_name]
print(f"x={x}")
print(f"y={y}")

# Encode the labels to numerical values if required (in our case we already have used a numerical value):

labelencoder = LabelEncoder()
labelencoder.fit(y)
y_encoded = labelencoder.transform(y)

print(f"y_encoded={y_encoded}")



# Split 1 (80% vs 20%)
x_train, x_validate_test, y_train, y_validate_test = train_test_split(x, 
y_encoded, test_size=0.20, random_state = 1)
# Split 2 (50% vs 50%)
x_test, x_validate, y_test, y_validate = train_test_split(x_validate_test, 
y_validate_test, test_size=0.50, random_state = 3)

# We split the data set into three parts, Training, Validation and Testing
# Training dataset: This dataset contains the samples to train the model. The weights and biases are learned with this data.
# Validation dataset: This dataset contains the samples to evaluate the model’s accuracy on unseen data. The dataset is used during training to indicate how well the model generalizes because it includes instances not included in the training dataset. However, since 
# this dataset is still used during training, we could indirectly influence the output model by fine-tuning some training hyperparameters.
# Test dataset: This dataset contains the samples for testing the model after training. Since the test dataset is not employed during training, it evaluates the final model without bias


model = tf.keras.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(len(f_names),)))

model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# In tinyML, monitoring the number of weights is crucial because it relates to the  program’s memory utilization. In our example we have 49 parameters which is relatively small

# We can now compile the model

model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])

#  loss: This is the loss function to minimize during training. The loss indicates how far 
# the predicted output is from the expected result, so the lower the loss, the better the 
# model. Cross-entropy isthe standard loss function for classification problems because 
# it produces faster training with better model generalization. For a binary classifier, we 
# should use binary_crossentropy.
# • optimizer: This is the algorithm used to update the network weights during training. 
# The optimizer mainly affects the training time. In our example, we use the widely adopted Adam optimizer.
# • metrics: This is the performance metric used to evaluate how well the model predicts 
# the output classes. We use accuracy, defined as the ratio between the number of correct 
# predictions and the total number of tests, as reported by the following equation: accuracy = number of correct predictions/total number of tests

# We train the model
NUM_EPOCHS=20
BATCH_SIZE=64
history = model.fit(x_train, y_train,
 epochs=NUM_EPOCHS, 
 batch_size=BATCH_SIZE, 
 validation_data=(x_validate,  y_validate))

model.save("./models/rain_forecast")