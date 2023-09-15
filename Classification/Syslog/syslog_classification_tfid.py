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
    "Internal build version date stamp (yyyy.mm.dd.vv) = 2023.06.21.01.kvm",
    "freerdp_abort_connect_context:freerdp_set_last_error_ex ERRCONNECT_CONNECT_CANCELLED [0x0002000B]",
    "publish_status: blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "publish_status: blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    "mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active",
    'mqtt: send_message: topic blackbox/sdkvm/deskvue/status/osd_sdkvm/connection/51/active this is not an "error" or a failure',
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

# For Random Forest classifiers, you can investigate feature importance to see which features (words or terms) were influential in the decision.
# This can give insights into why certain instances were misclassified.

feature_importances = loaded_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
important_features = sorted(
    zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
print(important_features[:10])  # Print the top N important features

# For the instance that was misclassified, inspect the predicted probability scores for each class.
# Most classifiers in scikit-learn, including RandomForestClassifier, have a predict_proba method that provides the probability scores for each class.
# This can help you understand how confident the model was in its prediction.
predicted_probs = loaded_model.predict_proba(new_data_features)
# this will show the predicted probabilities for each class, 0 and 1
print(predicted_probs)
