import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
# from keras.engine import Layer
import numpy as np
import csv
import pickle
import pandas as pd
import sys
from tensorflow.keras.layers import Dense
# https://github.com/tensorflow/text/blob/master/docs/tutorials/classify_text_with_bert.ipynb
# https://www.tensorflow.org/hub/tutorials


def analyze_data_object(obj,description=""):
    print("------------------------Object Analysis Begins--------------------------")
    # Check if it's a Series
    if isinstance(obj, pd.Series):
        print(f"Object {obj.name} is a pandas Series")
        print(f"Object Description {description}")
        print("Shape:", obj.shape)
        print("Type of elements:", type(obj.iloc[0]))
        print("First 3 elements:")
        print(obj.head(3))
    # Check if it's a DataFrame
    elif isinstance(obj, pd.DataFrame):
        print(f"Object {obj.name} is a pandas DataFrame")
        print(f"Object Description {description}")
        print("Shape:", obj.shape)
        print("Types of elements:")
        print(obj.dtypes)
        for column in obj.columns:
            print(f"Type of elements in column '{column}':", type(obj[column].iloc[0]))
        # Check type of elements in each row
        for index, row in obj.iterrows():
            print(f"Type of elements in row '{index}':", type(row.iloc[0]))
        print("First 3 elements:")
        print(obj.head(3))
    # Check if it's a NumPy array
    elif isinstance(obj, np.ndarray):
        print("Object is a NumPy array")
        print(f"Object Description {description}")
        print("Shape:", obj.shape)
        print("Type of elements:", obj.dtype)
        print("First 3 elements:")
        print(obj[:3])
    else:
        print("Unknown type")

    print("------------------------Object Analysis Ends--------------------------")




# Load datasets
with open('./data/spam_detection/train_data_elmo.pkl', 'rb') as f:
    train_x, train_y = pickle.load(f)

with open('./data/spam_detection/test_data_elmo.pkl', 'rb') as f:
    test_x, test_y = pickle.load(f)

# Define the file name
predictors_csv_file = "./data/spam_detection/predictors.csv"

# Read 'predictors' list from the CSV file
with open(predictors_csv_file, 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        predictors = row
        
         
print("TensorFlow version:", tf.__version__)
 
analyze_data_object(train_x,"train_x")
analyze_data_object(test_x,"test_x")
analyze_data_object(train_y,"train_y")
analyze_data_object(test_y,"test_y")

#sys.exit(0)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.elmo = hub.KerasLayer("https://tfhub.dev/google/elmo/3")
        super(ElmoEmbeddingLayer, self).build(input_shape)
 
    def call(self, x):
        return self.elmo(K.squeeze(K.cast(x, tf.string), axis=1))
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


def build_model():
    input_text = Input(shape=(1,), dtype=tf.string)
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = Dense(256, activation='relu')(embedding)
    pred = Dense(1, activation='sigmoid')(dense)
 
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
