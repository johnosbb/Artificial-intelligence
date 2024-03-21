import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import keras_nlp
from sklearn.model_selection import train_test_split
import sys

# https://www.machinelearningnuggets.com/text-classification-with-bert-and-kerasnlp/

df = pd.read_csv('./data/sentiment_analysis/sentiment.csv')
X = df['text']
y = df['sentiment']
X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

model_name = "bert_tiny_en_uncased_sst2"

# model_name: This parameter specifies the name of the BERT model variant you want to use. 
# In this case, it's 'bert_tiny_en_uncased_sst2', which refers to a specific variant of the 
# BERT model trained on English text for sentiment analysis (SST-2 dataset). 
# This variant is a "tiny" version of BERT, likely with fewer parameters compared to the original BERT model.

# num_classes: This parameter specifies the number of classes in your classification task. 
# In this case, it's set to 2, indicating a binary classification task (e.g., positive vs. negative sentiment).

# load_weights: This parameter specifies whether to load pre-trained weights for the BERT model.
# Setting it to True indicates that you want to use pre-trained weights.

# activation: This parameter specifies the activation function to use in the output layer of the classifier. In this case, it's set to 'sigmoid', which is appropriate for binary classification tasks, as it produces probabilities in the range [0, 1].

# Pretrained classifier.
classifier = keras_nlp.models.BertClassifier.from_preset(
    model_name,
    num_classes=2,
    load_weights = True,
    activation='sigmoid'
)

# The next step is to compile and train the model.
# The model's computational graph is compiled, which essentially sets up the model for training and evaluation.
# This involves linking the input and output layers, defining how the loss is computed,
# and specifying the optimization algorithm.
# Set the trainable parameter of the model to false so that you are not training the model from scratch. 
# Compiling the model sets up the necessary components for training, including loss function, 
# optimizer, metrics, and any additional optimizations like JIT compilation.
# Once compiled, the model is ready to be trained using the specified configurations.

# The objective is to use the pre-trained model and finetune it on your dataset,
# a process known as transfer learning.
classifier.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    jit_compile=True,
     metrics=["accuracy"],
)

# Access the backbone BERT model
backbone_model = classifier.backbone


# Access backbone programatically (e.g., to change `trainable`).
backbone_model.trainable = False


"""

Varying the number of layers to fine-tune in a BERT-based model can impact the model's performance,
training time, and generalization ability.
Here are some reasons why you might want to vary the number of layers you fine-tune:

Computational Resources: Fine-tuning all layers of a pre-trained BERT model can be computationally expensive,
especially if the model has a large number of layers and parameters.
By fine-tuning only a subset of layers, you can reduce the computational cost of training, making it feasible 
to train the model on hardware with limited resources.

Overfitting: Fine-tuning too many layers of a pre-trained BERT model on a small dataset can lead to overfitting.
By fine-tuning only a smaller number of layers, you can reduce the risk of overfitting,
as the model has fewer parameters to adapt to the training data.

Transferability of Knowledge: The lower layers of a pre-trained BERT model capture more generic features
of the language, such as syntax and grammar, which are likely to be useful across different
tasks and domains. Fine-tuning these layers may help the model adapt to the specifics 
of the target task or domain while still benefiting from the general language understanding
encoded in the pre-trained weights.

Task Complexity: For simpler tasks, such as sentiment analysis or text classification, 
fine-tuning fewer layers of the BERT model may be sufficient to achieve good performance.
On the other hand, for more complex tasks that require understanding of nuanced language semantics, 
fine-tuning more layers or even fine-tuning the entire model may be necessary to capture task-specific 
features effectively.

Domain-Specific Knowledge: In some cases, domain-specific knowledge may be encoded in the higher layers
of a pre-trained BERT model. Fine-tuning these layers allows the model to leverage this knowledge and 
adapt more effectively to tasks in that domain.

"""

# Determine the number of layers in the backbone model
num_layers = len(backbone_model.layers)

print("Number of layers in the backbone model:", num_layers)


# We can determine how many layers to finetune. In this case we Fine-tune only the last 10 layers
num_fine_tune_layers = 10
for layer in backbone_model.layers[-num_fine_tune_layers:]:
    layer.trainable = True
    

# Determine the current pooling layer
pooling_layer_name = "pooled_dense"  # Name of the pooling layer in BERT
# Find the pooling layer in the backbone model
pooling_layer = backbone_model.get_layer(pooling_layer_name)
print(f"pooling_layer= {pooling_layer}")

# Assuming 'pooling_layer' refers to the 'pooled_dense' layer
activation_function = pooling_layer.activation

print("Activation function of the pooling layer:", activation_function)
    
sys.exit()

# Fit again.
classifier.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), batch_size=32)

# Evaluating the model on the test set gives us an accuracy of 87% which is not bad 
# considering that you have used the tiny version of the BERT model.

classifier.evaluate(X_test, y_test,batch_size=32)

# Predict two new examples.
classifier.predict(["What an amazing movie!", "A total waste of my time."])


# You can also make the results more interpretable by passing the predictions through the class names 
# of the training data. Here is an example with a sample from the test set:

print(list(X_test)[10])
class_names = ["negative","postive"]
scores = classifier.predict([list(X_test)[10]])
scores
f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } percent confidence."
# https://www.machinelearningnuggets.com/text-classification-with-bert-and-kerasnlp/

# In the previous example, you trained a BERT model by passing raw strings. Notice that we didn't perform the standard NLP processing, such as:

# Removing punctuations
# Removing stop words
# Creating vocabulary
# Converting the text to a numerical computation
# All these were done by the model automatically. However, in some cases, you may want more control over that process. KerasNLP provides BertPreprocessor for this purpose. Every model has its preprocessor class. For this illustration, load BertPreprocessor with a sequence length of 128.

preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    model_name,
    sequence_length=128,
)

# Manually map this preprocessor to the training and testing set.
# Convert the data to a tf.data format to make this possible. Notice the use of:

# cache to cache the dataset. Pass a file name to this function if your dataset can't fit into memory.
# AUTOTUNE to automatically configure the batch size.

training_data = tf.data.Dataset.from_tensor_slices(([X_train], [y_train]))
validation_data = tf.data.Dataset.from_tensor_slices(([X_test], [y_test]))

train_cached = (
    training_data.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)
test_cached = (
    validation_data.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
)

# Pretrained classifier.
classifier = keras_nlp.models.BertClassifier.from_preset(
    model_name,
    preprocessor=None,
    num_classes=2,
    load_weights = True,
    activation='sigmoid'
)
classifier.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(),
    jit_compile=True,
     metrics=["accuracy"],
)
classifier.fit(train_cached, validation_data=test_cached,epochs=10)


# You can run some predictions on new data by first passing it through the BERT preprocessor to ensure 
# that it's in the format the model expects.

test_data = preprocessor([list(X_test)[10]])
print(list(X_test)[10])
scores = classifier.predict(test_data)
scores
f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } percent confidence." 

