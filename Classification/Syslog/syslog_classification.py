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
    "Jul 11 16:38:47 snuc-sdkvm app.py: publish_status: blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "Jul 11 16:38:47 snuc-sdkvm app.py: mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "Jul 11 16:38:47 snuc-sdkvm app.py: publish_status: blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "Jul 11 16:38:47 snuc-sdkvm app.py: mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "Jul 11 16:38:47 snuc-sdkvm app.py: mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active this is not an error or a failure",
    "Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153488]: freerdp_check_fds() failed - 0",
    "Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153488]: A thing has failed",
    "Jul 11 16:38:47 snuc-sdkvm bb_kvm_client[153488]: This thing is an indication of failure "
]
new_data_features = vectorizer.transform(new_data)
predictions = loaded_model.predict(new_data_features)

# You can use 'predictions' to get the predicted labels for the new data
print(predictions)
