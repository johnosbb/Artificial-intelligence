import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.sparse import issparse

# Specify the values to be treated as missing
missing_values = ["", "NA", "N/A", "NaN"]


file_path = './data/predictive_maintenance.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, na_values=missing_values)


features = ['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_Nm','tool_wear_min']
X = df[features]
y = df['target']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       
# Create an XGBoost model
model = XGBClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report_str)


