import tensorflow as tf

# Load the full TensorFlow model (Keras format or SavedModel format)
# For Keras model:
model = tf.keras.models.load_model('./models/preventive_forecast.keras')

# Alternatively, if you're using SavedModel format:
# model = tf.keras.models.load_model('./models/preventive_forecast_saved_model')

# Convert the model to TensorFlow Lite format without quantization (float32 precision)
converter = tf.lite.TFLiteConverter.from_keras_model(model)  # Use this if it's a Keras model

# If it's a SavedModel format, you can use the following:
# converter = tf.lite.TFLiteConverter.from_saved_model('./models/preventive_forecast_saved_model')

# **Disable quantization** by not applying any optimizations
# (This keeps the model in float32 precision)
tflite_model = converter.convert()

# Save the converted TFLite model to a file
tflite_model_path = './models/preventive_forecast_float32.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model successfully converted to TFLite format without quantization: {tflite_model_path}")
