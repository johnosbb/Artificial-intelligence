import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Load the ELMo model from TensorFlow Hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Define some example sentences
sentences = [
    "I love TensorFlow",
    "ELMo is awesome",
    "TensorFlow Hub makes it easy"
]

# Define the input layer for the sentences
input_text = Input(shape=(), dtype=tf.string)


# Convert the sentences to a tensor of strings
sentences_tensor = tf.constant(sentences)
print(f"sentences_tensor = {sentences_tensor}")

# Generate embeddings for the sentences
embeddings = elmo.signatures["default"](text=sentences_tensor)["elmo"]
print(embeddings)
