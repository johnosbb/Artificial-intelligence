import numpy as np
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)



data_a = np.array([300, 455, 350, 560, 700, 800, 200, 250])
labels = np.array([455, 350, 560, 700, 800, 200, 250, 300])

print(f"data_a : {data_a} Shape {data_a.shape}")
data_a = np.reshape(data_a, (8, 1))

inputs = keras.layers.Input(shape=(1, ))

x = keras.layers.Dense(40, activation='relu')(inputs)
output = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.models.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(data_a,labels, epochs=10, steps_per_epoch=4, verbose=1)

print("Ran Successfully")