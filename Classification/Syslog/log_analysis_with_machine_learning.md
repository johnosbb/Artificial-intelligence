# Log Analysis with Machine Learning


## Introduction

Many embedded systems product logs. These logs can provide valuable insights into the functioning of the system. They can provide information on patterns of behaviour and critically, information on errors and potential points of failure in the system.

This project is an attempt to explore how machine learning can be used to analyse system log files. I will be analysing a syslog file from an embedded system. The first task will be to try to predict indicators of failure or error. I have traditionally done this type of analysis using regular expressions and other pattern matching techniques. This approach requires constant update and tweaking. I hope, by using a machine learning approach, to develop a model that can accurately predict errors and system failures using a learned context. This would eliminate or greatly reduce the need to iterative tweaking of the tool.


## Preparing the data

The first stage of this process involves transforming the log data into a format that makes processing and classification more efficient.

The syslog data takes the following form:

```txt
Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153484]: Client terminated by remote command 
Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153482]: Client terminated by remote command 
Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153490]: Stopping client stats thread
Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153484]: Stopping client stats thread
Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153484]: Client stats thread cancelled
Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153490]: Client stats thread cancelled
```

To process this data I reformat the data into a pandas data frame with the following structure:

```python
    df = pd.DataFrame({
        "Date/Time": date_time_list,
        "Application": application_list,
        "Detail": detail_list,
        "Label": labels_list
    })
```

The "Detail" field will hold the main body of the log message and it is this field we will focus on when generating the target features.
There is some additional processing on some other lines to further target the main detail of the log entry.


```python
import re
import pandas as pd


# Specify the path to the input and output files
syslog_file_path = './data/Syslog/syslog'
filtered_syslog_file_path = './data/Syslog/syslog.cvs'


def check_for_exclusion_pattern(line):
    keywords_pattern = r'"error": "none"'
    keyword_match = re.search(
        keywords_pattern, line, re.IGNORECASE)
    if keyword_match:
        return True
    else:
        return False


def convert_syslog_to_dataframe(filepath):
    # Define a regular expression pattern to extract the information
    # \w{3} matches exactly three word characters (letters or digits). This corresponds to the month abbreviation like "Jul."
    # \d{2}:\d{2}:\d{2} matches the time in the format "HH:MM:SS," where HH represents the hours, MM represents the minutes, and SS represents the seconds.
    # (\S+): This part of the regular expression captures the next non-space sequence of characters, which corresponds to the application name (e.g., "app.py" in your syslog lines).
    # \S+ matches one or more non-space characters.
    pattern = r'(\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\S+) (.*)'

    # Create empty lists to store extracted data
    date_time_list = []
    application_list = []
    detail_list = []
    labels_list = []
    records_processed = 2
    # Extract data from syslog lines
    with open(syslog_file_path, 'r') as file:
        for line in file:
            if(records_processed == 975):
                print("Found target")
            line = line.rstrip()
            match = re.match(pattern, line)
            if match:
                keywords_pattern = r'error|failed|failure|ERRCONNECT|\berr\b'
                keyword_match = re.search(
                    keywords_pattern, line, re.IGNORECASE)
                if keyword_match:
                    print(
                        f"The string on line {records_processed} contains the target keyword. {line}")
                    if(check_for_exclusion_pattern(line) == True):
                        label = 0
                    else:
                        label = 1
                else:
                    label = 0
                date_time = match.group(1)
                application = match.group(3)
                detail = match.group(4)
                date_time_list.append(date_time)

                if "[OSD]" in detail:
                    application = "OSD"
                    osd_pattern = r'\[OSD\] : \[.+?\] \[.+?\] - (.*)'
                    match = re.match(osd_pattern, detail)
                    num_groups = len(match.groups())
                    if match and (num_groups == 1):
                        detail_ammended = match.group(1)
                        detail_list.append(detail_ammended)
                    else:
                        print(
                            f"something went wrong on line {records_processed}: {detail}")
                        detail_list.append(detail)
                else:
                    detail_list.append(detail)
                application_list.append(application)
                labels_list.append(label)
                records_processed = records_processed + 1
            else:
                print(
                    f"Could not match or filter this line: [{line}] on line number {records_processed}")
    # Create a DataFrame
    print(f"Processed {records_processed} records")
    df = pd.DataFrame({
        "Date/Time": date_time_list,
        "Application": application_list,
        "Detail": detail_list,
        "Label": labels_list
    })
    return df


# def quick_test():
#     "Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[154037]: Program parameters:"


# Save the DataFrame
df = convert_syslog_to_dataframe(syslog_file_path)
df.to_csv(filtered_syslog_file_path, index=False)


```

The output of this process is a file syslog.cvs which stores the pandas data frame as a cvs file.


## Creating the Model

For the analysis of the data I initially tried a bag-of-words (BoW) vectorizer. This is a simple and fundamental technique used for text analysis and feature extraction. It's a way to represent text data, such as sentences or documents, as numerical vectors that can be used in machine learning algorithms.

```python
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Import joblib

filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model.joblib'
vectorizer_filename = './data/vectorizer.joblib'  # Choose a filename


df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
# Labels indicating whether a particular event of interest occurred (1 for occurrence, 0 for non-occurrence)
labels = df["Label"]

# Create a bag-of-words (BoW) vectorizer
vectorizer = CountVectorizer()

# Convert preprocessed log lines to feature vectors
feature_vectors = vectorizer.fit_transform(details)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, labels, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest classifier in this example)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file

joblib.dump(clf, model_filename)
joblib.dump(vectorizer, vectorizer_filename)
```

In this approach we are predicting whether a specific event or condition occurred based on the content of the log lines.
To be more precise:

- The labels list contains binary labels for each log line:
- 1 indicates that a specific event or condition of interest occurred in that log line.
- 0 indicates that the event or condition did not occur in that log line.

For instance, in the context of system log analysis, the events or conditions of interest could be various system states, errors, warnings,
or other significant events. These events or conditions are typically defined based on the log data and the specific goals of the analysis.

The goal of the machine learning model is to learn patterns in the log data that are indicative of the occurrence
(or non-occurrence) of these events or conditions. Once trained, the model can predict whether an event or condition
is likely to occur in new, unseen log lines.

The model and vectorizer are stored for later use using joblib.

For this particular approach we get an accuracy of 0.99 based on the test and training data.


## Testing the Model with Unseen Data

We now create a simple test program to see how likely the model is at predicting unseen errors (log entries it has not previously seen)

```python

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Import joblib

model_filename = './data/Syslog/random_forest_model.joblib'
# Load the vectorizer from a separate file (if you saved it separately)
vectorizer_filename = './data/vectorizer.joblib'


# Load the model from the file
loaded_model = joblib.load(model_filename)


# Access the vectorizer from the loaded_model
vectorizer = joblib.load(vectorizer_filename)


# Make predictions using the loaded model
new_data = [
    "Internal build version date stamp (yyyy.mm.dd.vv) = 2023.06.21.01.device",
    "freerdp_abort_connect_context:freerdp_set_last_error_ex ERRCONNECT_CONNECT_CANCELLED [0x0002000B]",
    "Jul 11 16:38:47 my-device app.py: publish_status: system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: publish_status: system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active this is not an error or a failure",
    "Jul 11 16:38:47 my-device client[153488]: freerdp_check_fds() failed - 0",
    "Jul 11 16:38:47 my-device client[153488]: Something has failed",
    "Jul 11 16:38:47 my-device client[153488]: This thing is an indication of failure "
]
new_data_features = vectorizer.transform(new_data)
predictions = loaded_model.predict(new_data_features)

# You can use 'predictions' to get the predicted labels for the new data
print(predictions)
```

In this code I introduce some previously seen log entries and some new entries with content that may indicate an error.

The program makes the following predictions:
[0 1 0 0 0 0 0 1 1 0]

We can see that for the previously unseen data:

```txt
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active this is not an error or a failure", - 0
    "Jul 11 16:38:47 my-device client[153488]: freerdp_check_fds() failed - 0",                                                                                   - 1
    "Jul 11 16:38:47 my-device client[153488]: Something has failed",                                                                                             - 1
    "Jul 11 16:38:47 my-device client[153488]: This thing is an indication of failure "                                                                           - 0
```

That it had a 50% success rate. The first entry is deliberately ambiguous so its classification of this as a negative could be considered a mis-classification.

Clearly the model struggled to predict the new log entries and the previously predicted accuracy.

We would like to understand why we achieved such poor results so we can look a little more closely at the workings of the model and consider some of the factors that may have contributed to the performance.

## Improving our Model

The misclassification of certain log lines in our test data can be attributed to several factors. Let's break down why these lines might be misclassified and how we can investigate and potentially improve the model's performance:

- Class Imbalance: If our dataset is imbalanced, meaning there are significantly more samples of one class (e.g., non-occurrence) than the other (e.g., occurrence), the model may be biased towards the majority class. We can check the balance between our classes in the training data and consider techniques like oversampling or undersampling to balance the dataset. This is quite likely to be the cause of some of the performance issues because the log files we used are from a real world system and they are likely to contain few errors which would mean the data is very imbalanced.
- Feature Extraction: In our code, we used a Bag-of-Words (BoW) vectorizer to convert log lines into feature vectors. BoW may not capture the context and semantics of log lines effectively, especially if the log messages are complex and contain important information beyond simple word counts. We could consider using more advanced techniques like TF-IDF or word embeddings (e.g., Word2Vec or FastText) to represent log messages better.
- Model Choice: Random Forest is a versatile algorithm, but it may not be the best choice for all types of text classification tasks. We could experiment with other algorithms like Support Vector Machines (SVM), Gradient Boosting, or deep learning models (e.g., LSTM or CNN) to see if they perform better on our dataset.
- Feature Engineering: We could explore additional features that may help improve classification accuracy. For log data, we could consider features like log message length, presence of specific keywords, or patterns in the log messages.
- Hyperparameter Tuning: We could look at fine-tune the hyperparameters of our classifier using techniques like cross-validation and grid search to find the best hyperparameters for our model.
- Error Analysis: We could examine the misclassified examples in detail. Print out the misclassified log lines and their predicted labels to gain insights into why the model is making errors. We could look for common patterns or keywords in the misclassified lines that the model is not capturing.
- Additional Data: We could collect more labeled data, especially examples that the model is misclassifying. A larger and more diverse dataset may help the model learn better patterns.
- Preprocessing: We might be able to review our preprocessing steps, such as removing timestamps and other noise to ensure that these steps are not inadvertently removing important information that the model needs to make accurate predictions.
- Exclusion Patterns: We should pay attention to exclusion patterns and how they are affecting the labels. Make sure that the exclusion patterns are correctly identifying cases where log lines should be labeled as 0 (non-occurrence).
- Ensemble Methods: We might consider using ensemble methods like stacking or boosting to combine the predictions of multiple models. This can often lead to improved performance.


### Hyperparameter Tuning

I used a Random Forest Classifier algorithm with default parameters. We could consider tuning some of these parameters. We have already noted that the data is likely imbalanced so a good candidate for tuning is the class_weight. We can set class_weight to 'balanced' to automatically adjust the weights of classes inversely proportional to their frequencies.

```python

# n_estimators: The number of trees in the forest. Increasing the number of trees can improve model performance up to a point. However, more trees also mean longer training times.
# max_depth: The maximum depth of each tree in the forest. Increasing max_depth can make the trees more complex and potentially capture more intricate patterns in the data. Be cautious not to set it too high to avoid overfitting.
# min_samples_split: The minimum number of samples required to split an internal node. Increasing this parameter can make the tree less likely to split, which can reduce overfitting.
# min_samples_leaf: The minimum number of samples required to be at a leaf node. Similar to min_samples_split, increasing this parameter can regularize the tree.
# max_features: The number of features to consider when looking for the best split. You can experiment with different values, such as  (sqrt(n_features)), 'log2' (log2(n_features)), or an integer representing the number of features.
# class_weight: If your dataset is imbalanced (which is common in anomaly detection tasks like log analysis), you can set class_weight to 'balanced' to automatically adjust the weights of classes inversely proportional to their frequencies.
# bootstrap: Whether or not to use bootstrap samples when building trees. Setting it to False can be useful if you want to disable bootstrapping.
# random_state: Set a specific random seed for reproducibility.
# Train a machine learning model (Random Forest classifier in this example)
# Train a machine learning model (Random Forest classifier) with modified hyperparameters
clf = RandomForestClassifier(
    # n_estimators=100,  # Increase the number of trees
    # max_depth=None,    # Allow trees to grow until fully developed
    # min_samples_split=2,  # Reduce the minimum samples required to split
    # min_samples_leaf=1,   # Allow smaller leaf nodes
    # max_features='sqrt',  # Consider all features for splitting
    class_weight='balanced',  # Adjust class weights for imbalanced data
    # bootstrap=True,   # Use bootstrapped samples
    random_state=42   # Set a specific random seed for reproducibility
)

```

When we rerun our classification code we get the same result as before:

```txt
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active this is not an error or a failure", - 0
    "Jul 11 16:38:47 my-device client[153488]: freerdp_check_fds() failed - 0",                                                                                   - 1
    "Jul 11 16:38:47 my-device client[153488]: Something has failed",                                                                                             - 1
    "Jul 11 16:38:47 my-device client[153488]: This thing is an indication of failure "                                                                           - 0
```

#### Automating our Hypertuning using Gridsearch

The RandomForestClassifier is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. There are a number of parameters that we can vary while attempting to tune the model [see](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

We can automate the hypertuning  by using grid search. Grid search is a technique used for hyperparameter tuning in machine learning. Hyperparameters are settings for a machine learning algorithm that are not learned from the data but are set prior to training. They can significantly affect the performance of a model, and finding the best combination of hyperparameters is essential for building an effective model.

Grid search works by exhaustively searching through a predefined set of hyperparameters to find the combination that produces the best model performance. 

We can recode our solution to use gridsearch.

```python
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import GridSearchCV

filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model_tfid.joblib'
vectorizer_filename = './data/vectorizer_tfid.joblib'

# Preprocess log lines (remove timestamps and other noise)
df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
labels = df["Label"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(details)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, labels, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(
    random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the grid search to your training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# Use the best estimator for predictions
y_pred = best_estimator.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(best_estimator, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

```

This gives us the best performing parameters:

```txt
Best Hyperparameters:
{
    'class_weight': 'balanced',
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 100
}

Accuracy: 0.99

```

### Trying an Alternative Feature Extraction Approach

We can try replacing our Bag of Words with TF-IDF (TF-IDF (Term Frequency-Inverse Document Frequency) to capture the importance of words in the log lines and may improve the model's performance, especially if certain keywords are more informative for your classification task. TF-IDF takes into account both the frequency of words in a document and their importance in the corpus, which can be more effective than simple word counts.

```python
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model_tfid.joblib'
vectorizer_filename = './data/vectorizer_tfid.joblib'

# Preprocess log lines (remove timestamps and other noise)

df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
labels = df["Label"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()  # Use TfidfVectorizer instead of CountVectorizer

# Convert preprocessed log lines to TF-IDF feature vectors
feature_vectors = vectorizer.fit_transform(details)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, labels, test_size=0.2, random_state=42)


# n_estimators: The number of trees in the forest. Increasing the number of trees can improve model performance up to a point. However, more trees also mean longer training times.
# max_depth: The maximum depth of each tree in the forest. Increasing max_depth can make the trees more complex and potentially capture more intricate patterns in the data. Be cautious not to set it too high to avoid overfitting.
# min_samples_split: The minimum number of samples required to split an internal node. Increasing this parameter can make the tree less likely to split, which can reduce overfitting.
# min_samples_leaf: The minimum number of samples required to be at a leaf node. Similar to min_samples_split, increasing this parameter can regularize the tree.
# max_features: The number of features to consider when looking for the best split. You can experiment with different values, such as  (sqrt(n_features)), 'log2' (log2(n_features)), or an integer representing the number of features.
# class_weight: If your dataset is imbalanced (which is common in anomaly detection tasks like log analysis), you can set class_weight to 'balanced' to automatically adjust the weights of classes inversely proportional to their frequencies.
# bootstrap: Whether or not to use bootstrap samples when building trees. Setting it to False can be useful if you want to disable bootstrapping.
# random_state: Set a specific random seed for reproducibility.
# Train a machine learning model (Random Forest classifier in this example)
# Train a machine learning model (Random Forest classifier) with modified hyperparameters
clf = RandomForestClassifier(
    # n_estimators=100,  # Increase the number of trees
    # max_depth=None,    # Allow trees to grow until fully developed
    # min_samples_split=2,  # Reduce the minimum samples required to split
    # min_samples_leaf=1,   # Allow smaller leaf nodes
    # max_features='sqrt',  # Consider all features for splitting
    class_weight='balanced',  # Adjust class weights for imbalanced data
    # bootstrap=True,   # Use bootstrapped samples
    random_state=42   # Set a specific random seed for reproducibility
)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(clf, model_filename)
joblib.dump(vectorizer, vectorizer_filename)


```


We now modify the classification code to use this new feature extraction approach.

```python
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
import joblib  # Import joblib

model_filename = './data/Syslog/random_forest_model_tfid.joblib'
# Load the vectorizer from a separate file (if you saved it separately)
vectorizer_filename = './data/vectorizer_tfid.joblib'

# Load the model from the file
loaded_model = joblib.load(model_filename)

# Access the vectorizer from the loaded_model
vectorizer = joblib.load(vectorizer_filename)

# Make predictions using the loaded model
new_data = [
    "Internal build version date stamp (yyyy.mm.dd.vv) = 2023.06.21.01.device",
    "freerdp_abort_connect_context:freerdp_set_last_error_ex ERRCONNECT_CONNECT_CANCELLED [0x0002000B]",
    "Jul 11 16:38:47 my-device app.py: publish_status: system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: publish_status: system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active",
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active this is not an error or a failure",
    "Jul 11 16:38:47 my-device client[153488]: freerdp_check_fds() failed - 0",
    "Jul 11 16:38:47 my-device client[153488]: Something has failed",
    "Jul 11 16:38:47 my-device client[153488]: This thing is an indication of failure "
]
new_data_features = vectorizer.transform(new_data)
predictions = loaded_model.predict(new_data_features)

# You can use 'predictions' to get the predicted labels for the new data
print(predictions)



```

With this approach we get the following results:


```txt
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active this is not an error or a failure", - 1
    "Jul 11 16:38:47 my-device client[153488]: freerdp_check_fds() failed - 0",                                                                                   - 0
    "Jul 11 16:38:47 my-device client[153488]: Something has failed",                                                                                             - 0    
```

We saw no significant improvement with this approach. So we will now look at something to more specifically address the lack of balance in the data.

### Addressing Class Imbalance

We now introduce some oversampling to try to address the data imbalance.  If we have a class imbalance issue (i.e., class 0 has significantly more instances than class 1), we can try oversampling the minority class (class 1) or undersampling the majority class (class 0). This can balance the dataset and potentially make the model more sensitive to class 1 instances.

#### Random Oversampling with Custom Ratios

In random oversampling, we randomly duplicate instances from the minority class to balance the class distribution. To place more emphasis on class 1, we can adjust the oversampling ratio. Instead of simply replicating instances until class balance is achieved, we can set a custom oversampling ratio that specifies how many times you want to oversample class 1 relative to class 0. For example, if we set a ratio of 3:1, it means we will oversample class 1 three times as much as class 0. This way, more emphasis is placed on class 1 during the oversampling process.

```python
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Import RandomOverSampler from imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
import joblib

filtered_syslog_file_path = './data/Syslog/syslog.cvs'
model_filename = './data/Syslog/random_forest_model_tfid_with_oversampling.joblib'
vectorizer_filename = './data/vectorizer_tfid_with_oversampling.joblib'

# Preprocess log lines (remove timestamps and other noise)
df = pd.read_csv(filtered_syslog_file_path)
details = df["Detail"]
labels = df["Label"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()  # Use TfidfVectorizer instead of CountVectorizer

# Convert preprocessed log lines to TF-IDF feature vectors
feature_vectors = vectorizer.fit_transform(details)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    feature_vectors, labels, test_size=0.2, random_state=42)

# Random oversampling
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_oversampled, y_train_oversampled = oversampler.fit_resample(
    X_train, y_train)

# Train a machine learning model (Random Forest classifier) with modified hyperparameters
clf = RandomForestClassifier(
    random_state=42
)
clf.fit(X_train_oversampled, y_train_oversampled)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(clf, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

```

Following this change we get the these results:


```txt
    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active this is not an error or a failure", - 1
    "Jul 11 16:38:47 my-device client[153488]: freerdp_check_fds() failed - 0",                                                                                   - 0
    "Jul 11 16:38:47 my-device client[153488]: Something has failed",                                                                                             - 0    
```

### Examining the Features


We saw no improvement so we need to understand which features are influential in the decision making.
We can print out the top ten most important features.

```python
# For Random Forest classifiers, you can investigate feature importance to see which features (words or terms) were influential in the decision.
# This can give insights into why certain instances were misclassified.
feature_importances = loaded_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
important_features = sorted(
    zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
print(important_features[:10])  # Print the top N important features
```


```txt
[
 ('failed', 0.09232203342472536),
 ('callback', 0.06438864072439934),
 ('setting', 0.05950584768240038),
 ('error', 0.05669028186091807),
 ('glfw', 0.054403262326712734),
 ('freerdp_check_fds', 0.04946882013851324),
 ('vtrfx_check_transport_events', 0.03925084404923942),
 ('checking', 0.03046707450390108),
 ('connection', 0.025609590594164334),
 ('disconnected', 0.02385549238546408)
]
```

For the instance that was misclassified, we inspect the predicted probability scores for each class.
Most classifiers in scikit-learn, including RandomForestClassifier, have a predict_proba method that provides the probability scores for each class.
This can help us understand how confident the model was in its prediction.

```python
predicted_probs = loaded_model.predict_proba(new_data_features)
# this will show the predicted probabilities for each class, 0 and 1
print(predicted_probs)
```


| Log Entry | Probability of Class 0 | Probability of Class 1 | Actual Classification |
| --------- | ---------------------- | ---------------------- | ---------------------- |
|    "Internal build version date stamp (yyyy.mm.dd.vv) = 2023.06.21.01.device" |  1.  |  0.  | 0 |
|    "freerdp_abort_connect_context:freerdp_set_last_error_ex ERRCONNECT_CONNECT_CANCELLED [0x0002000B]" |  0.41 | 0.59 | 1 |
|    "Jul 11 16:38:47 my-device app.py: publish_status: system/device/deskvue/status/osd_device/connection/51/active" | 0.95 | 0.05 | 0 |
|    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active" |  0.96 | 0.04 |  0 |
|    "Jul 11 16:38:47 my-device app.py: publish_status: system/device/deskvue/status/osd_device/connection/51/active" | 0.95 | 0.05 |  0 |
|    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active" | 0.96 | 0.04 |  0 |
|    "Jul 11 16:38:47 my-device app.py: mqtt: send_message: topic system/device/deskvue/status/osd_device/connection/51/active this is not an error or a failure" |  0.87 | 0.13 |  0 |
|    "Jul 11 16:38:47 my-device client[153488]: freerdp_check_fds() failed - 0" | 0.3 |  0.7 |  1 |
|    "Jul 11 16:38:47 my-device client[153488]: Something has failed" |  0.54 | 0.46 |  0 |
|    "Jul 11 16:38:47 my-device client[153488]: This thing is an indication of failure " | 0.87 | 0.13 |  0 |


