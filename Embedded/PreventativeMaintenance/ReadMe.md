# ReadMe

## Generate a Synthetic Dataset

### generate_synthetic_maintenance_data.py

This script generates a synthetic dataset designed to demonstrate how machine learning models, such as those built with TensorFlow, can predict motor failures based on sensor data. The dataset is created with the goal of simulating real-world scenarios where various parameters—such as RPM, temperature, vibration, and current—are monitored to assess the likelihood of a motor failing.

The code starts by creating random data for four key features: RPM (revolutions per minute), temperature, vibration, and current. Each feature is generated using normal distributions centered around realistic values, with specific ranges of variation. For example, RPM values are centered around 1600 with a standard deviation of 200, while temperature values are centered around 24°C with a standard deviation of 5.

Next, the script defines several thresholds for what constitutes a failure condition for each feature. For instance, if the temperature exceeds 30°C, or if the RPM drops below 1500, it is considered a failure condition. The script then evaluates each data point against these thresholds. If a failure condition is met, the likelihood of the motor actually failing is increased, introducing a greater chance of failure. Conversely, if no failure conditions are met, the likelihood of a failure is reduced.

The generated data is stored in a pandas DataFrame, which includes both the sensor readings and a failure indicator. The dataset is then saved to a CSV file for further use. Additionally, the script prints out some basic statistics about the class distribution (i.e., the number of failures versus non-failures) and the current working directory where the dataset is saved.

This exercise is meant for educational purposes, providing a controlled environment to understand how sensor data can be used to train predictive models. By creating this dataset, we can explore how machine learning techniques, such as those available in TensorFlow, can be applied to predict potential motor failures based on various input parameters.

## Train Model and Evaluate

### predictive_maintenance_model.py

This script is designed to train a machine learning model using TensorFlow to predict motor failures based on various sensor data. The process involves several key steps, including data loading, preprocessing, model building, training, and evaluation.

First, the script loads a dataset from a CSV file that contains sensor readings and a failure indicator. It separates the data into features (such as RPM, temperature, vibration, and current) and the target variable (motor failures). Before feeding the data into a model, it calculates and displays the mean and standard deviation of each feature to understand their distributions.

The script then splits the dataset into training, validation, and test sets. To address class imbalance, it optionally applies the Synthetic Minority Over-sampling Technique (SMOTE) to balance the class distribution in the training set. This technique generates synthetic samples for the minority class to help the model learn better from imbalanced data.

Next, the script scales the features using standard normalization (Z-score normalization) and visualizes the distributions of key variables if specified. It then defines a machine learning model using TensorFlow’s Keras API. Depending on the configuration, the model could be a simple one or a more complex neural network with dropout layers to prevent overfitting.

The model is compiled with appropriate loss and optimizer functions and trained using the training data. Early stopping is employed to halt training if the model’s performance on the validation set doesn’t improve, thus preventing overfitting.

After training, the model is saved for future use and evaluated on the test set to determine its accuracy. Predictions are made, and an adjusted threshold is applied to classify the results. A confusion matrix is generated to assess the model’s performance, and various evaluation metrics like accuracy, precision, recall, and F-score are computed to understand the model’s effectiveness in predicting motor failures.

Overall, this script demonstrates a comprehensive approach to building a predictive model for motor failure, from data preparation and preprocessing to model training and evaluation. The goal is to provide insights into how machine learning can be used to predict equipment failures based on sensor data.

## Converting the Data to TFLite form

### convert_to_tflite_with_quantization.py

This script is designed to convert a trained TensorFlow model into a TensorFlow Lite (TFLite) format for deployment on mobile or embedded devices. It starts by loading a pre-trained model and the associated dataset. The data is then split into training, validation, and test sets to prepare for the quantization process.

Quantization is a technique used to reduce the precision of model parameters, converting them from 32-bit floating-point numbers to 8-bit integers. This helps to decrease the model size and increase inference speed, which is crucial for deploying models on resource-constrained devices.

The script includes a function to generate a representative dataset used during the quantization process. This dataset helps the converter to understand the data distribution and apply appropriate scaling to the quantized model. It then configures the TFLite converter to use this representative dataset, applies the quantization optimizations, and sets the operation types to ensure the model is compatible with 8-bit integer quantization. Finally, the script converts the model, saves the quantized TFLite model to a file, and prints the size of the quantized model in bytes.

## A Note on Scale and Zero-Point

### Scale and Zero Point Formulas

When converting floating-point values to integers, two important parameters are the **scale** and the **zero point**.

#### 1. Scale Calculation

The **scale** is calculated using the formula:

```
scale = (max_float - min_float) / (max_int - min_int)
```

where:

- `max_float` is the maximum floating-point value.
- `min_float` is the minimum floating-point value.
- `max_int` is the maximum integer value representable in the target integer type.
- `min_int` is the minimum integer value representable in the target integer type.

#### 2. Zero Point Calculation

The **zero point** is calculated using the formula:

```
zero_point = round(min_float / scale)
```

where:

- `min_float` is the minimum floating-point value.

### Example

For a floating-point range from -1.0 to 1.0 and an 8-bit signed integer range from -128 to 127:

1. **Calculate Scale:**

```
scale = (1.0 - (-1.0)) / (127 - (-128)) = 2.0 / 255 ≈ 0.00784
```

2. **Calculate Zero Point:**

```
zero_point = round(-1.0 / 0.00784) = round(-127.56) = -128
```
