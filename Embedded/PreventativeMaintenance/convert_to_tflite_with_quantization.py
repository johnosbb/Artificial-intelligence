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
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = './data/predictive_maintenance_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Split the data into features and target
X = data.drop('Motor Fails', axis=1)  # Features
y = data['Motor Fails']  # Target

# Split the data into training and temp (validation + test) sets
x_train, x_validate_test, y_train, y_validate_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Further split the temp set into validation and test sets
x_test, x_validate, y_test, y_validate = train_test_split(x_validate_test, y_validate_test, test_size=0.50, random_state=3)


# Before converting the TensorFlow model to TensorFlow Lite (TFLite), we chose to normalize the data used for generating the representative dataset. 
# This decision was made because the input features in the original dataset have widely varying ranges—RPM is measured in the thousands, while vibration values are much smaller, often less than 1.
# Such discrepancies in the scale of input features could lead to issues during quantization, as the model may struggle to handle the diverse ranges effectively. 
# To mitigate this, we applied Z-score normalization to standardize the data, ensuring that each feature contributes equally during the quantization process.
# By doing so, we aimed to improve the model’s performance and accuracy when deployed on resource-constrained embedded devices.
# The original code has been left in place, but commented out

# This function generates a small subset of the test data in the appropriate format to be used as representative data
#  during the quantization process.
# This version uses the original data (i.e non-normalised)
# def representative_data_gen():
#   for i_value in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
#     i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
#     yield [i_value_f32]

# Scale the features using Z-score normalization, this is to prepare the data for the alternative of normalised data in the representative
# dataset

scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(x_train)
X_validate_scaled = scaler.transform(x_validate)
X_test_scaled = scaler.transform(x_test)


# Define the representative dataset generator, but with normalised data to try and address the imbalance in the scale of RPM and Vibrartion data

def representative_data_gen():
    # Generate representative normalized data for calibration
    for i_value in tf.data.Dataset.from_tensor_slices(X_test_scaled).batch(1).take(100):
        i_value_f32 = tf.dtypes.cast(i_value, tf.float32)
        yield [i_value_f32]


# Convert the exported SavedModel to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model("./models/preventive_forecast_saved_model")

converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert to TFLite model
tflite_model_quant = converter.convert()
model_path = "./models/preventive_forecast.tflite"
open(model_path, "wb").write(tflite_model_quant)

# Display the size of the quantized model
size_tfl_model = len(tflite_model_quant)
print(f"Quantized Model size in bytes:  {size_tfl_model} bytes")

# Load TFLite model to extract scale and zero-point
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get details about the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print quantization details for debugging
#  Remember that the model’s input uses the per-tensor quantization schema, so all 
# input features must be quantized with the same scale and zero point
input_scales = input_details[0]['quantization'][0]  # Array of scales if available
input_zero_points = input_details[0]['quantization'][1]  # Array of zero points if available
output_scale, output_zero_point = output_details[0]['quantization']


print(f"Input Tensor Scales: {input_scales}, Zero Points: {input_zero_points}")
print(f"Output Tensor - Scale: {output_scale}, Zero Point: {output_zero_point}")
