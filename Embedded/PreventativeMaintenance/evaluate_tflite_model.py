import numpy as np
import tensorflow as tf

# Function to normalize input data (Z-score normalization)
def normalize_input(input_data, mean_values, std_values):
    return (input_data - mean_values) / std_values

# Function to de-normalize input data back to the original scale (for debugging or alternate inference paths)
def denormalize_input(normalized_data, mean_values, std_values):
    return (normalized_data * std_values) + mean_values

# Function to convert normalized float32 data to int8 (using scale and zero point)
def float32_to_int8(input_data, scale, zero_point):
    int8_data = (input_data / scale) + zero_point
    return np.clip(int8_data, -128, 127).astype(np.int8)

# Load the TFLite model
model_path = "./models/preventive_forecast.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get quantization parameters
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]

print(f"Input Scale: {input_scale}, Input Zero Point: {input_zero_point}")
print(f"Output Scale: {output_scale}, Output Zero Point: {output_zero_point}")

# Mean and std values for normalization (obtained during training)
mean_values = np.array([1603.866, 24.354, 0.120, 3.494], dtype=np.float32)
std_values = np.array([195.843, 4.987, 0.020, 0.308], dtype=np.float32)

# Prepare input data (RPM, Temperature, Vibration, Current)
input_data = np.array([[1700, 25.5, 0.135, 3.75]], dtype=np.float32)  # Example values

# Step 1: Normalize the input data using Z-score normalization
normalized_input = normalize_input(input_data, mean_values, std_values)
print(f"Normalized Input: {normalized_input}")

# Step 2: Quantize the normalized float32 input to int8
int8_input_data = float32_to_int8(normalized_input, input_scale, input_zero_point)
print(f"Quantized Int8 Input Data: {int8_input_data}")

# Step 3: Set the tensor and run inference
interpreter.set_tensor(input_details[0]['index'], int8_input_data)
interpreter.invoke()

# Step 4: Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Raw Output Tensor (Quantized Int8): {output_data}")

# Step 5: Dequantize the output to float32
dequantized_output = (output_data - output_zero_point) * output_scale
print(f"Dequantized Output (Float32): {dequantized_output}")

# Step 6: Interpret the result
failure_prob = dequantized_output.item()  # Extract the scalar value
print(f"Motor Failure Prediction Probability: {failure_prob:.6f}")

if failure_prob > 0.5:
    print("Motor Failure Detected")
else:
    print("No Motor Failure Detected")
