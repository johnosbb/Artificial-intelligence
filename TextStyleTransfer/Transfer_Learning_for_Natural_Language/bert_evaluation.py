import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import keras_nlp
from sklearn.model_selection import train_test_split
import pickle
import sys

"""
Pretrained BERT Model Initialization: We start by loading a pretrained BERT model. This model has been trained on a large corpus of text and has learned contextual representations of words.

Building a Classifier on Top of BERT: We build a classifier on top of the pretrained BERT model. This classifier will take the contextual embeddings produced by BERT and use them to perform a specific task, in this case, sentiment analysis.

Compiling the Classifier: We compile the classifier by specifying the loss function, optimizer, and metrics to be used during training.

Setting the Model to Trainable: By default, the BERT model loaded is frozen, meaning its weights won't be updated during training. However, we set it to trainable, allowing its weights to be fine-tuned during training on our specific dataset.

Training the Classifier: We train the classifier on our dataset using the fit method. This step involves passing our training data through the model, computing the loss, and updating the weights of the model using backpropagation.

Evaluating the Model: We evaluate the trained model on a separate test set to assess its performance on unseen data.

Making Predictions: We use the trained model to make predictions on new examples. These predictions are based on the learned representations of the input text.

Interpreting Predictions: To make the predictions more interpretable, we map the predicted scores (probabilities) back to the class names (in this case, sentiment labels) and calculate the confidence of the prediction.
"""

LOAD_SPAM_DATASETS=True
FIT_MODEL=True
LOAD_SENTIMENT_CSV=False
SAVE_SPAM_CSV=True
REPLACE_MODEL_LAYERS=False
DETERMINE_LAYER_ACTIVATION_FUNCTION=False
CONTROL_TUNABLE_LAYERS=False

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


def drop_rows(df):
    # Count the number of NaN values before dropping
    num_nan_before = df.isna().sum().sum()
    # Drop rows with NaN values
    df = df.dropna()
    # Count the number of NaN values after dropping
    num_nan_after = df.isna().sum().sum()
    # Calculate the number of rows dropped
    num_dropped = num_nan_before - num_nan_after
    return num_dropped

if LOAD_SPAM_DATASETS:
    # Load datasets
    with open('./data/spam_detection/train_data_bert.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f) # load a single dimension numpy data array

    with open('./data/spam_detection/test_data_bert.pkl', 'rb') as f:
        X_test, y_test = pickle.load(f)

    # X_train.name="X_train"
    # y_train.name="y_train"
    # X_test.name="X_test"
    # y_test.name="y_test" 
    analyze_data_object(X_train,"X_train from pickle")
    analyze_data_object(y_train,"y_train from pickle")
    analyze_data_object(X_test,"X_test from pickle")
    analyze_data_object(y_test,"y_test from pickle")



    # Flatten the list of lists
    X_train_flat = [item for sublist in X_train for item in sublist]
    X_train_1d = np.array(X_train_flat)
    # Create a pandas Series
    X_train = pd.Series(X_train_1d)

    X_test_flat = [item for sublist in X_test for item in sublist]
    X_test_1d = np.array(X_test_flat)
    # Create a pandas Series
    X_test = pd.Series(X_test_1d)

    analyze_data_object(X_train,"X_train_series after flat")
    # Combine X_train and y_train into a single DataFrame
    df_train = pd.concat([X_train, pd.Series(y_train)], axis=1) # pd.Series(y_train) converts y_train from a numpy array to a series
    df_train.columns = ['text', 'classification']  # Naming the columns
    df_test = pd.concat([X_test, pd.Series(y_test)], axis=1)
    df_test.columns = ['text', 'classification']  # Naming the columns
    dropped = drop_rows(df_train)
    print(f"Dropped {dropped} train rows.")
    dropped = drop_rows(df_test)
    print(f"Dropped {dropped} test rows.")
    # Generate random classification values (0 or 1)
    if SAVE_SPAM_CSV:
        # classification = np.random.randint(0, 2, size=len(X_train))
        # X_train_with_classification = X_train.to_frame(name='text')
        # X_train_with_classification['classification'] = y_train
        df_train.to_csv('./data/spam_detection/X_train.csv', index=True)
        df_test.to_csv('./data/spam_detection/X_test.csv', index=True)
    y_train = tf.keras.utils.to_categorical(df_train['classification'], num_classes=2) # used to convert class labels into one-hot encoded vectors. This function is often used in classification tasks, especially when dealing with categorical data.
    y_test = tf.keras.utils.to_categorical(df_test['classification'], num_classes=2)
    class_names = ["spam", "not_spam"]  # Assuming it's a typo and should be "positive" instead of "postive"
    data_sample = ["Please end funs immediately", "to whom it may concern."]

if LOAD_SENTIMENT_CSV:
    # based on
    # https://www.machinelearningnuggets.com/text-classification-with-bert-and-kerasnlp/

    #df = pd.read_csv('./data/sentiment_analysis/sentiment_13.csv')

    df = pd.read_csv('./data/sentiment_analysis/X_train_with_sentiment.csv')
    df = df.dropna()

    # Verify the shape after filtering out missing values
    print("Shape after filtering:", df.shape)
    # print(f"df: {df} type(df) {type(df)}  df {df.shape}")
    X = df['text'] # when you extract a single column from a DataFrame using df['column_name'], it is returned as a pandas Series.
    y = df['sentiment'] # a DataFrame is essentially a collection of Series, where each Series represents a column. Therefore, extracting a single column results in a Series containing the values of that column.
    print(f"Using Sentiment Analaysis")
    X.name="X from CSV"
    y.name="y from CSV"
    analyze_data_object(X)
    analyze_data_object(y)

    class_names = ["negative", "positive"]  # Assuming it's a typo and should be "positive" instead of "postive"

    X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2) # used to convert class labels into one-hot encoded vectors. This function is often used in classification tasks, especially when dealing with categorical data.
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    print(f"Using Sentiment Analaysis - after one-hot encoding")
    X_train.name="X_train_csv"

    X_test.name="X_test_csv"

    analyze_data_object(X_train)
    analyze_data_object(y_train)
    analyze_data_object(X_test)
    analyze_data_object(y_test)
    data_sample = ["What an amazing movie!", "A total waste of my time."]


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

# Determine the number of layers in the backbone model
num_layers = len(backbone_model.layers)
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

if CONTROL_TUNABLE_LAYERS:
    # We can determine how many layers to finetune. In this case we Fine-tune only the last 10 layers
    num_fine_tune_layers = 10
    for layer in backbone_model.layers[-num_fine_tune_layers:]:
        layer.trainable = True
    

if DETERMINE_LAYER_ACTIVATION_FUNCTION:    
    # Determine the current pooling layer
    pooling_layer_name = "pooled_dense"  # Name of the pooling layer in BERT
    # Find the pooling layer in the backbone model
    pooling_layer = backbone_model.get_layer(pooling_layer_name)
    print(f"pooling_layer= {pooling_layer}")
    # Assuming 'pooling_layer' refers to the 'pooled_dense' layer
    activation_function = pooling_layer.activation
    print("Activation function of the pooling layer:", activation_function)



if REPLACE_MODEL_LAYERS: # This is not currently working
    def build_model(max_seq_length):
        in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]
        print(f"bert_inputs = {bert_inputs}  Type of item: {type(bert_inputs)}")
        # Print the content of bert_output to understand its structure
        # Extract BERT features
        bert_output = backbone_model(bert_inputs)
        print("Content of bert_output:")
        print(f"bert_output {bert_output} Type of item: {type(bert_output)}")

        # Iterate over the content of bert_output
        for key, value in bert_output.items():
            print(f"Tensor name: {key}")
            print(f"Tensor shape: {value.name}")
            print(f"Tensor sparse: {value.sparse}")
            print(f"Tensor shape: {value.shape}")


        # # just extract BERT features, don't fine-tune
        # bert_output = BertLayer(n_fine_tune_layers=0)(bert_inputs)
        # # train dense classification layer on top of extracted features
        # dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
        # pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

        # model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        # model.summary()

        #return model
    
    build_model(256)

if FIT_MODEL:
    # Fit again.
    classifier.fit(x=X_train, y=y_train, validation_data=(X_test,y_test), batch_size=32)
    #sys.exit()

    # Evaluating the model on the test set gives us an accuracy of 87% which is not bad 
    # considering that you have used the tiny version of the BERT model.

    classifier.evaluate(X_test, y_test,batch_size=32)

    # Predict two new examples.
    classifier.predict(data_sample)


    # You can also make the results more interpretable by passing the predictions through the class names 
    # of the training data. Here is an example with a sample from the test set:

    scores = classifier.predict([list(X_test)[10]])
    result = f"{class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} percent confidence."
    print(result)

    # In the previous example, you trained a BERT model by passing raw strings. 
    # Notice that we didn't perform the standard NLP processing, such as:

    # Removing punctuations
    # Removing stop words
    # Creating vocabulary
    # Converting the text to a numerical computation
    # All these were done by the model automatically.
    # However, in some cases, you may want more control over that process.
    # KerasNLP provides BertPreprocessor for this purpose. 
    # Every model has its preprocessor class. For this illustration, load BertPreprocessor with a sequence length of 128.

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

    scores = classifier.predict(test_data)
    result = f"{class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} percent confidence."
    print(result)
