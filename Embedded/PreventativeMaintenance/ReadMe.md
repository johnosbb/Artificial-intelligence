# ReadMe

## Generate a Synthetic Dataset

### generate_synthetic_maintenance_data.py

This script marks the first stage in the process of developing a machine learning model that can be trained and evaluated on a PC, and later optimized to run efficiently on resource-constrained embedded devices like an Arduino board. The goal is to predict motor failures based on sensor data, which simulates real-world scenarios where motors are continuously monitored to ensure reliability. By generating and training the model on a PC, we can then fine-tune and optimize it for embedded systems, ensuring it operates effectively even in limited environments.

The script generates a synthetic dataset designed to simulate various conditions under which a motor might fail, using sensor readings for key features like RPM (revolutions per minute), temperature, vibration, and current. Each feature is generated using random values from normal distributions that mimic realistic operating conditions. For example, RPM values are centered around 1600 with a standard deviation of 200, while temperature values hover around 24°C with a standard deviation of 5.

The main benefit of generating this dataset from a set of specific rules is that these rules can be used to measure the success of the model once it's deployed on embedded devices. The script defines several failure thresholds for the features—such as RPM dropping below 1500 or temperature exceeding 30°C. It then calculates the likelihood of a motor failure based on whether any of these conditions are met. If a failure condition occurs, the likelihood of failure increases, simulating real-world risk. Conversely, normal operating conditions reduce the likelihood of failure.

The data, along with a failure indicator, is stored in a pandas DataFrame and saved to a CSV file for later use in training and evaluating machine learning models. Additionally, the script provides basic statistics about the class distribution (i.e., the number of failures vs. non-failures) and displays the current working directory where the dataset is stored.

This exercise is meant for educational purposes, providing a controlled environment to understand how sensor data can be used to train predictive models. By creating this dataset, we can explore how machine learning techniques, such as those available in TensorFlow, can be applied to predict potential motor failures based on various input parameters.

## Train Model and Evaluate

### predictive_maintenance_model.py

This script is designed to train a machine learning model using TensorFlow to predict motor failures based on various sensor data. The process involves several key steps, including data loading, preprocessing, model building, training, and evaluation.

First, the script loads a dataset from a CSV file that contains sensor readings and a failure indicator. It separates the data into features (such as RPM, temperature, vibration, and current) and the target variable (motor failures). Before feeding the data into a model, it calculates and displays the mean and standard deviation of each feature to understand their distributions.

The script then splits the dataset into training, validation, and test sets. To address class imbalance, it optionally applies the Synthetic Minority Over-sampling Technique (SMOTE) to balance the class distribution in the training set. This technique generates synthetic samples for the minority class to help the model learn better from imbalanced data.

Next, the script scales the features using standard normalization (Z-score normalization) and visualizes the distributions of key variables if specified. It then defines a machine learning model using TensorFlow’s Keras API. Depending on the configuration, the model could be a simple one or a more complex neural network with dropout layers to prevent overfitting.

The model is compiled with appropriate loss and optimizer functions and trained using the training data. Early stopping is employed to halt training if the model’s performance on the validation set doesn’t improve, thus preventing overfitting.

After training, the model is saved for future use and evaluated on the test set to determine its accuracy. Predictions are made, and an adjusted threshold is applied to classify the results. A confusion matrix is generated to assess the model’s performance, and various evaluation metrics like accuracy, precision, recall, and F-score are computed to understand the model’s effectiveness in predicting motor failures.

It’s important to note that this dataset is somewhat contrived, and as such, we should not be overly concerned with the accuracy and performance statistics of the resulting model. Our primary focus here is the process: building the model on a resource-rich platform like a PC, and then understanding the steps required to optimize and deploy it on an edge device with limited computational resources.

## Converting the Data to TFLite form

### convert_to_tflite_with_quantization.py

This script is designed to convert a trained TensorFlow model into a TensorFlow Lite (TFLite) format for deployment on mobile or embedded devices. It starts by loading a pre-trained model and the associated dataset. The data is then split into training, validation, and test sets to prepare for the quantization process.

Quantization is a technique used to reduce the precision of model parameters, converting them from 32-bit floating-point numbers to 8-bit integers. This helps to decrease the model size and increase inference speed, which is crucial for deploying models on resource-constrained devices.

The script includes a function to generate a representative dataset used during the quantization process. This dataset helps the converter to understand the data distribution and apply appropriate scaling to the quantized model. Before converting the TensorFlow model to TensorFlow Lite (TFLite), we chose to normalize the data used for generating the representative dataset. This decision was made because the input features in the original dataset have widely varying ranges—RPM is measured in the thousands, while vibration values are much smaller, often less than 1. Such discrepancies in the scale of input features could lead to issues during quantization, as the model may struggle to handle the diverse ranges effectively. To mitigate this, we applied Z-score normalization to standardize the data, ensuring that each feature contributes equally during the quantization process. By doing so, we aimed to improve the model’s performance and accuracy when deployed on resource-constrained embedded devices.

The script finally configures the TFLite converter to use this representative dataset, applies the quantization optimizations, and sets the operation types to ensure the model is compatible with 8-bit integer quantization. Finally, the script converts the model, saves the quantized TFLite model to a file, and prints the size of the quantized model in bytes.

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
