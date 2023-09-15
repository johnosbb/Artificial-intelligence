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
    "Internal build version date stamp (yyyy.mm.dd.vv) = 2023.06.21.01.kvm",
    "freerdp_abort_connect_context:freerdp_set_last_error_ex ERRCONNECT_CONNECT_CANCELLED [0x0002000B]",
    "publish_status: blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "publish_status: blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active this is not an error or a failure",
    "freerdp_check_fds() failed - 0",
    "A thing has failed",
    "This thing is an indication of a failed system because of the error",
    "This thing is an indication of a failure because of an error",
    "Setting glfw error callback"
]
new_data_features = vectorizer.transform(new_data)
predictions = loaded_model.predict(new_data_features)

# You can use 'predictions' to get the predicted labels for the new data
print(predictions)
