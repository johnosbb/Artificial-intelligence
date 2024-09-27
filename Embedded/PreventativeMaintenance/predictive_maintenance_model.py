import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
import sklearn
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

SHOW_DISTRIBUTIONS = True
SIMPLE_MODEL = True
LINEAR_REGRESSION = True
USE_SMOTE=True
NUMBER_OF_FEATURES=8 #32
NUM_EPOCHS = 20
BATCH_SIZE = 64

# Load the dataset
file_path = './data/predictive_maintenance_dataset.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Split the data into features and target
X = data.drop('Motor Fails', axis=1)  # Features
y = data['Motor Fails']  # Target

# Calculate the mean and standard deviation for each feature before scaling and balancing
feature_means = X.mean()
feature_stds = X.std()

# Print the statistical summary for each feature
print("Statistical Summary of Features Before Scaling and Balancing:")
for feature in X.columns:
    print(f"{feature} - Mean: {feature_means[feature]:.3f}, Standard Deviation: {feature_stds[feature]:.3f}")



# Count the occurrences of each class (0 or 1)
counts = data['Motor Fails'].value_counts()

# Calculate the ratio for each class
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

if USE_SMOTE:
    # Adjusted SMOTE parameters
    smote = SMOTE(k_neighbors=3, sampling_strategy=0.5, random_state=42, n_jobs=-1)
    X_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
    # Check the distribution of the target variable after balancing
    print("Distribution of 'Motor Fails' after SMOTE:")
    print(y_train_balanced.value_counts())
else:
    X_train_balanced = x_train
    y_train_balanced = y_train.values  # Convert to NumPy array


# Convert the balanced and scaled data back to a DataFrame for visualization
X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=x_train.columns)

if SHOW_DISTRIBUTIONS:
    # Visualize RPM vs. Temperature for synthetic vs. real data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_train_balanced_df['RPM'], y=X_train_balanced_df['Temperature (°C)'], hue=y_train_balanced)
    plt.title('RPM vs. Temperature after SMOTE')
    plt.show()


# Check the distribution of the target variable after balancing
print("Distribution of 'Motor Fails' after SMOTE:")
print(y_train_balanced.value_counts())


# Scale the features using Z-score normalization
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_validate_scaled = scaler.transform(x_validate)
X_test_scaled = scaler.transform(x_test)


min_value = np.min(X_train_balanced_scaled)
max_value = np.max(X_train_balanced_scaled)

print(f"Minimum value: X_train_balanced_scaled:  {min_value}")
print(f"Maximum value: X_train_balanced_scaled: {max_value}")


# Convert back to DataFrame for visualization
X_train_balanced_scaled_df = pd.DataFrame(X_train_balanced_scaled, columns=X.columns)

if SHOW_DISTRIBUTIONS:
    # Plotting the main variables
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.histplot(X_train_balanced_scaled_df['RPM'], kde=True, color='blue')
    plt.title('RPM Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(X_train_balanced_scaled_df['Temperature (°C)'], kde=True, color='red')
    plt.title('Temperature Distribution')

    plt.subplot(2, 2, 3)
    sns.histplot(X_train_balanced_scaled_df['Vibration (g)'], kde=True, color='green')
    plt.title('Vibration Distribution')

    plt.subplot(2, 2, 4)
    sns.histplot(X_train_balanced_scaled_df['Current (A)'], kde=True, color='purple')
    plt.title('Current Distribution')

    plt.tight_layout()
    plt.show()

# Extract the input features (x - RPM, Temperature, Vibration, Current) and output labels (y - Motor Fails) from the dataset:
f_names = X_train_balanced_scaled_df.columns.values
x = X_train_balanced_scaled_df[f_names]
print(f"x={f_names}:{x}")

if SIMPLE_MODEL:
    model = tf.keras.Sequential([
    layers.Input(shape=(X_train_balanced_scaled.shape[1],)),
    layers.Dense(NUMBER_OF_FEATURES, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Single output neuron with sigmoid
])
else:
    # Define the model with increased complexity
    model = tf.keras.Sequential()
    model.add(layers.Dense(NUMBER_OF_FEATURES, activation='relu', input_shape=(len(f_names),)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

# Compile the model with class weights to handle imbalance

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
if USE_SMOTE:
    class_weights_dict = None # Remove class weights is using smote
else:    
    class_weights_dict = dict(enumerate(class_weights))
#class_weights_dict = {0: 1.0, 1: 1.0}  # Adjust weights based on class imbalance if required
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Train the model

history = model.fit(X_train_balanced_scaled, y_train_balanced,
                    epochs=NUM_EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    validation_data=(X_validate_scaled, y_validate),
                    class_weight=class_weights_dict, callbacks=[early_stopping])

# Save the model in native Keras format
model.save("./models/preventive_forecast.keras")
model.export("./models/preventive_forecast_saved_model")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Predict and adjust the threshold
y_test_pred = model.predict(X_test_scaled)
adjusted_threshold = 0.514  # Adjust this value as needed
y_test_pred = (y_test_pred > adjusted_threshold).astype("int32")

# Generate confusion matrix
cm = sklearn.metrics.confusion_matrix(y_test, y_test_pred)

index_names = ["Actual No Failure", "Actual Failure"]
column_names = ["Predicted No Failure", "Predicted Failure"]
df_cm = pd.DataFrame(cm, index=index_names, columns=column_names)
if SHOW_DISTRIBUTIONS:
    plt.figure(dpi=150)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
    plt.show()

# Calculate evaluation metrics
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
accur = (TP + TN) / (TP + TN + FN + FP)
precis = TP / (TP + FP)
recall = TP / (TP + FN)
f_score = (2 * recall * precis) / (recall + precis)
print("Accuracy: The proportion of correct predictions (both true positives and true negatives) out of all predictions : ", round(accur, 3))
print("Recall: The proportion of actual positive cases that were correctly identified. :  ", round(recall, 3))
print("Precision: The proportion of predicted positive cases that are actually positive : ", round(precis, 3))
print("F-score: The harmonic mean of precision and recall : ", round(f_score, 3))
