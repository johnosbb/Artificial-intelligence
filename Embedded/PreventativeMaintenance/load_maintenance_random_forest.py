import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

SHOW_GRAPHS = False
USE_SMOTE = True

# Load the dataset
file_path = './data/predictive_maintenance_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Split the data into features and target
X = data.drop('Motor Fails', axis=1)  # Features
y = data['Motor Fails']  # Target

# Split the data into training and temp (validation + test) sets
x_train, x_validate_test, y_train, y_validate_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Further split the temp set into validation and test sets
x_test, x_validate, y_test, y_validate = train_test_split(x_validate_test, y_validate_test, test_size=0.50, random_state=3)

# Scale the features using Z-score normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_validate_scaled = scaler.transform(x_validate)
X_test_scaled = scaler.transform(x_test)

if USE_SMOTE:
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate Random Forest Classifier
y_test_pred_rf = rf_model.predict(X_test_scaled)
print("Random Forest Classifier Performance:")
print(classification_report(y_test, y_test_pred_rf))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred_rf)
index_names = ["Actual No Failure", "Actual Failure"]
column_names = ["Predicted No Failure", "Predicted Failure"]
df_cm = pd.DataFrame(cm, index=index_names, columns=column_names)

if SHOW_GRAPHS:
    plt.figure(dpi=150)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
    plt.show()


rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
importances = rf.feature_importances_

minority_class_data = data[data['Motor Fails'] == 1]
print(minority_class_data.describe())

if SHOW_GRAPHS:
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=X.columns)
    plt.title('Feature Importances')
    plt.show()