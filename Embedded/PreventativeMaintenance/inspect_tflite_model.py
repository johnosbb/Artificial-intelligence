import tensorflow as tf

# Load the quantized TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='./models/preventive_forecast.tflite')
interpreter.allocate_tensors()

# Get information about the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loop over the details and print out scale and zero point
for i, detail in enumerate(input_details):
    scale = detail['quantization'][0]  # Scale value
    zero_point = detail['quantization'][1]  # Zero point
    print(f"Input Tensor {i}: Scale = {scale}, Zero Point = {zero_point}")

for i, detail in enumerate(output_details):
    scale = detail['quantization'][0]  # Scale value
    zero_point = detail['quantization'][1]  # Zero point
    print(f"Output Tensor {i}: Scale = {scale}, Zero Point = {zero_point}")
