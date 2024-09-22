import numpy as np
import tensorflow as tf

# Function to normalize input data
def normalize_input(input_data, mean_values, std_values):
    return (input_data - mean_values) / std_values

if __name__ == '__main__':
    # Load the non-quantized TFLite model
    model_path = "./models/preventive_forecast_float32.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print input and output details for debugging
    print(f"Input Details: {input_details}")
    print(f"Output Details: {output_details}")

    # Define mean and std values for normalization (replace with your actual values)
    mean_values = np.array([1603.866, 24.354, 0.120, 3.494], dtype=np.float32)
    std_values = np.array([195.843, 4.987, 0.020, 0.308], dtype=np.float32)

    # Prepare input data (RPM, Temperature, Vibration, Current)
    input_data = np.array([[1700, 25.5, 0.135, 3.75]], dtype=np.float32)  # Example values
    normalized_input = normalize_input(input_data, mean_values, std_values)  # Normalize the input

    # Debug: print normalized input data
    print("Normalized Input (float32):", normalized_input)

    # Set the input tensor (directly using float32 input)
    interpreter.set_tensor(input_details[0]['index'], normalized_input)

    # Run inference
    interpreter.invoke()

    # Get the output tensor (already in float32)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Debug: print raw output data
    print("Raw Output Data (float32):", output_data)

    # Since the output is already in float32, no dequantization is needed
    failure_prob = output_data.item()  # Get the scalar value from the output

    # Display the result
    print(f"Motor Failure Prediction Probability: {failure_prob:.6f}")
    if failure_prob > 0.5:
        print("Motor Failure Detected")
    else:
        print("No Motor Failure Detected")

