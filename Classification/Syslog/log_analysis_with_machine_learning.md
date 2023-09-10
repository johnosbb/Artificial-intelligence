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
