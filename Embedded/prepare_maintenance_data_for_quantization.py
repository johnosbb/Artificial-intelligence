import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
import sklearn
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping



# Load the dataset
file_path = './data/predictive_maintenance_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Split the data into features and target
X = data.drop('Motor Fails', axis=1)  # Features
y = data['Motor Fails']  # Target

# Split the data into training and temp (validation + test) sets
x_train, x_validate_test, y_train, y_validate_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Further split the temp set into validation and test sets
x_test, x_validate, y_test, y_validate = train_test_split(x_validate_test, y_validate_test, test_size=0.50, random_state=3)

# Quantization is a process used to reduce the precision of the numbers that represent a model's parameters,
#  often from 32-bit floating-point to 8-bit integers.
#  This reduces the model's size and increases inference speed, which is particularly important for deploying models on mobile or embedded devices.

# This function generates a small subset of the test data in the appropriate format to be used as representative data during the quantization process.
def representative_data_gen():
  for i_value in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100): # Batches the data one sample at a time, Takes 100 samples from the dataset. This subset will be used to "calibrate" the quantization process.
    i_value_f32 = tf.dtypes.cast(i_value, tf.float32) # Ensures that the input values are cast to 32-bit floating-point format, which might be required before quantization.
    yield [i_value_f32] # The function is a generator, yielding one batch of data at a time.

# Convert the exported SavedModel to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model("./models/preventive_forecast_saved_model")

converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen) # We set the representative from our generation function above dataset for post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT] # We set the optimization strategy for the conversion process. We apply a general optimization strategy, which typically includes quantization to reduce the model size and improve performance on edge devices.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # We set of operations (ops) that the converted model should use. This restricts the operations to those that support 8-bit integer quantization. This ensures that the model can be fully quantized to int8, which is necessary for the most efficient execution on hardware that supports int8 operations.
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


tflite_model_quant = converter.convert()
open("./models/preventive_forecast.tflite", "wb").write(tflite_model_quant)



