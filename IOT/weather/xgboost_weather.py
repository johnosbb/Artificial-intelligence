import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, precision_score
from scipy.sparse import issparse



# Specify the values to be treated as missing
missing_values = ["", "NA", "N/A", "NaN"]


file_path = './data/weather_data.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, na_values=missing_values)



# Convert non-numeric values to NaN for all columns using map
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
# Drop rows with NaN values in any column
df = df.dropna()


print("Number of NaN values in 'rain':", df['rain'].isna().sum())
print("Number of infinite values in 'rain':", np.isinf(df['rain']).sum())


# Assuming your dataset is stored in a DataFrame named 'df'
X = df.drop(columns=['rain'])  # Features
y = df['rain']  # Target variable
# Convert y to a DataFrame
y = pd.DataFrame(y, columns=['rain'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X: {type(X)} is_sparce: {issparse(X)}\n {X}\n")
print(f"y: {type(y)} is_sparce: {issparse(y)}\n {y}\n")
print(f"y_train: {type(y_train)} is_sparce: {issparse(y_train)}\n {y_train}\n")
print(f"y_test: {type(y_test)} is_sparce: {issparse(y_test)}\n {y_test}\n")

#  "sparse" refers to a representation where most of the elements are zero,
# and only the non-zero elements are explicitly stored. Sparse representations
# are used to efficiently handle datasets where the majority of values are zero,
# which is common in various domains, including text processing,
# network analysis, and certain scientific applications.

       
# Create an XGBoost model
model = XGBRegressor()

# Fit the model to the training data
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Convert y_test DataFrame to a Series
y_test_series = y_test['rain']


mape = np.mean(np.abs((y_test_series - y_pred) / y_test_series)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')

# Mean Squared Error (MSE):

# MSE measures the average of the squared differences between the actual and predicted values.
# The smaller the MSE, the better the model's performance. A lower MSE indicates that, on average, the model's predictions are closer to the actual values.
# The MSE value itself doesn't have a fixed scale. It's relative to the scale of your target variable. In your case, the MSE is 1302.34.
# Mean Absolute Percentage Error (MAPE):

# MAPE expresses the average absolute percentage difference between the actual and predicted values as a percentage.
# A lower MAPE indicates better model performance. A MAPE of 37.25% means, on average, the model's predictions are off by approximately 37.25% relative to the actual values.
# MAPE is a percentage, making it more interpretable across different datasets and scales.