import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Layer
from keras.models import Model

num_train_samples = 1400
num_test_samples = 600
num_features = 600

train_x = np.random.rand(num_train_samples, num_features)
train_x2 = np.random.rand(num_train_samples, num_features)
train_y = np.random.randint(2, size=(num_train_samples, 1))
train_y2 = np.random.randint(2, size=(num_train_samples, 1))

test_x = np.random.rand(num_test_samples, num_features)
test_x2 = np.random.rand(num_test_samples, num_features)
test_y = np.random.randint(2, size=(num_test_samples, 1))
test_y2 = np.random.randint(2, size=(num_test_samples, 1))

input1_shape = (num_features,)
input2_shape = (num_features,)

sent2vec_vectors1 = Input(shape=input1_shape, name="vector1")
sent2vec_vectors2 = Input(shape=input2_shape, name="vector2")

class ConcatenateLayer(Layer):
    def call(self, inputs, axis=0):
        return tf.concat(inputs, axis=axis)

combined = ConcatenateLayer()([sent2vec_vectors1, sent2vec_vectors2], axis=-1)

dense1 = Dense(512, activation='relu')(combined)
dense1 = Dropout(0.3)(dense1)
output1 = Dense(1, activation='sigmoid', name='classification1')(dense1)
output2 = Dense(1, activation='sigmoid', name='classification2')(dense1)

model = Model(inputs=[sent2vec_vectors1, sent2vec_vectors2], outputs=[output1, output2])

model.compile(
    loss={'classification1': 'binary_crossentropy', 'classification2': 'binary_crossentropy'},
    optimizer='adam',
    metrics=['accuracy','accuracy']
)

history = model.fit(
    [train_x, train_x2],
    [train_y, train_y2],
    validation_data=([test_x, test_x2], [test_y, test_y2]),
    epochs=10,
    shuffle=True)