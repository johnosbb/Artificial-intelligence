import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

SHOW_DISTRIBUTIONS = False
USE_SMOTE = True

# Load the dataset
file_path = './data/predictive_maintenance_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Count the occurrences of each class (0 or 1)
counts = data['Motor Fails'].value_counts()
ratios = counts / len(data)

print("Class distribution (counts):")
print(counts)

print("\nClass distribution (ratios):")
print(ratios)

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

if SHOW_DISTRIBUTIONS:
    # Convert back to DataFrame for visualization
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

    # Plotting the main variables
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(X_train_scaled_df['RPM'], kde=True, color='blue')
    plt.title('RPM Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(X_train_scaled_df['Temperature'], kde=True, color='red')
    plt.title('Temperature Distribution')

    plt.subplot(2, 2, 3)
    sns.histplot(X_train_scaled_df['Vibration'], kde=True, color='green')
    plt.title('Vibration Distribution')

    plt.subplot(2, 2, 4)
    sns.histplot(X_train_scaled_df['Current'], kde=True, color='purple')
    plt.title('Current Distribution')

    plt.tight_layout()
    plt.show()

# Train a simple Logistic Regression model
model = LogisticRegression(class_weight='balanced')  # 'balanced' automatically adjusts weights based on class distribution
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_test_pred = model.predict(X_test_scaled)

# y_test_pred_adjusted = (y_test_pred > 0.61).astype(int)
# y_test_pred == y_test_pred_adjusted

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
index_names = ["Actual No Failure", "Actual Failure"]
column_names = ["Predicted No Failure", "Predicted Failure"]
df_cm = pd.DataFrame(cm, index=index_names, columns=column_names)

if SHOW_DISTRIBUTIONS:
    plt.figure(dpi=150)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
    plt.show()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f_score = f1_score(y_test, y_test_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F-score: {f_score:.3f}")
