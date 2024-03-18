import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
# from keras.engine import Layer
import numpy as np
import csv
import pickle


# https://github.com/tensorflow/text/blob/master/docs/tutorials/classify_text_with_bert.ipynb
# https://www.tensorflow.org/hub/tutorials


# Load datasets
with open('./data/spam_detection/train_data.pkl', 'rb') as f:
    train_x, train_y = pickle.load(f)

with open('./data/spam_detection/test_data.pkl', 'rb') as f:
    test_x, test_y = pickle.load(f)

# Define the file name
predictors_csv_file = "./data/spam_detection/predictors.csv"

# Read 'predictors' list from the CSV file
with open(predictors_csv_file, 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        predictors = row
        
         
print("TensorFlow version:", tf.__version__)
 
print("Shape of train_x:", train_x.shape)
print("Shape of test_x:", test_x.shape)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.elmo = hub.load("https://tfhub.dev/google/elmo/3")
        super(ElmoEmbeddingLayer, self).build(input_shape)
 
    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default')['default']
        return result
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

    



def build_model(): 
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embedding = ElmoEmbeddingLayer(trainable=True)(input_text)  # Set trainable=True here
    dense = layers.Dense(256, activation='relu')(embedding)
    pred = layers.Dense(1, activation='sigmoid')(dense)
 
    model = Model(inputs=input_text, outputs=pred)
 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
  
    return model

 
# Build and fit
model = build_model()
model.fit(train_x,
          train_y,
          validation_data=(test_x, test_y),
          epochs=5,
          batch_size=32)    