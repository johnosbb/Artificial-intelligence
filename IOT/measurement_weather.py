
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot,norm
import seaborn as sns
import numpy.ma as ma
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error, r2_score

# Specify the values to be treated as missing
missing_values = ["", "NA", "N/A", "NaN"]


file_path = './data/weather_data.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, na_values=missing_values)
# Convert the 'rain' column to numeric, handling non-numeric values
# df['rain'] = pd.to_numeric(df['rain'], errors='coerce')
# Convert blank spaces to numeric for all columns
df = df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
                 
                 
# Display the DataFrame
print(df)

# year,month,meant,maxtp,mintp,mnmax,mnmin,rain,gmin,wdsp,maxgt,sun


# Calculate the variance of the 'rain' column
# The variance is a squared measure of how much each individual daily rainfall value deviates from the mean.
# The larger the variance, the more spread out the data points are from the mean. 
# The unit of variance is squared, so it doesn't have the same unit as the original data.
# If you want a measure in the same unit as your original data, you can take the square root of the variance to get the standard deviation.
df['temperature_variance'] = df['meant'].var()





# Calculate mean and Z-score
mean_rainfall = df['rain'].mean()
std_dev_rainfall = df['rain'].std()

df['mean_rainfall'] = mean_rainfall
df['z-score'] = (df['rain'] - mean_rainfall) / std_dev_rainfall

min_temp = df['meant'].min()
max_temp = df['meant'].max()


# Specify the window size for the rolling analysis (e.g., 3 days)
window_size = 3

# Perform a rolling mean calculation
df['rolling_mean'] = df['rain'].rolling(window=window_size).mean()


# Specify the overlap size for a hopping window
overlap = 2



# Adjust the index to create overlapping windows
df['hopping_mean'] = df['rolling_mean'].shift(-overlap)


# Display the DataFrame
print(df)


# Plotting the bar chart
df.plot(x='year', y='rain', kind='bar', figsize=(10, 6), legend=False)

# Adjusting x-axis labels to every nth item (e.g., every 2nd item)
n = 100
plt.xticks(range(0, len(df), n), df['year'].iloc[::n], rotation=45) # df['year'].iloc[::n]: This selects every nth year from the 'year' column of the DataFrame.

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.title('Rainfall Data Over Time')


# Save the plot as a PNG file
plt.savefig('rainfall_plot.png')

# Display the plot
plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(df['meant'], df['rain'])
plt.xlabel('Temperature')
plt.ylabel('Rainfall (mm)')
plt.title('Scatter Plot: Temperature vs. Rainfall')

# Save the plot as a PNG file
plt.savefig('rainfall_temperature_plot.png')
# Display the plot
plt.show()


# Plotting the bubble chart
plt.scatter(df['meant'], df['maxgt'], s=df['rain'], alpha=0.5)

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Bubble Chart for Temperature, Rain and Grass Temperature')

# Save the plot as a PNG file
plt.savefig('bubble_chart.png')

# Display the plot
plt.show()





plt.figure(figsize=(10, 6))

# Extract the 'rain' column
rain_data = df['rain']

# Remove non-finite values (NaN and infinite)
rain_data = rain_data[np.isfinite(rain_data)]


# Create a Q-Q plot for the 'rain' data
probplot(rain_data, dist="norm", plot=plt)
plt.title("Q-Q Plot for Rain Data - Test for Normal Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('q_q_plot.png')

# Show the plot
plt.show()


# Plot a histogram
plt.figure(figsize=(10, 6))
sns.histplot(rain_data, bins=20, kde=False, color='blue', stat='density', element='step')

# Fit a normal distribution to the data
mu, std = norm.fit(rain_data)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
# Save the plot as a PNG file
plt.savefig('histogram.png')
plt.plot(x, p, 'k', linewidth=2) # x: This is the array of x-values (independent variable) where the line will be plotted. p: This is the array of y-values (dependent variable) specifying the height of the line at each x-value. 'k': This is a format string specifying the color and line style of the plot. In this case, 'k' stands for black. Matplotlib uses a variety of format strings to control the appearance of the plot, and 'k' is a shorthand for a solid black line.

# Draw a vertical line at the mean
plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label='Mean') # draw a vertical dashed line at the mean of the distribution (mu). 


# Calculate points at ±3 standard deviations
number_of_std_devs = 3
lower_limit = mu - number_of_std_devs * std
upper_limit = mu + number_of_std_devs * std

# Draw vertical lines at ±3 standard deviations
plt.axvline(lower_limit, color='green', linestyle='dashed', linewidth=2, label='-3 Std Dev')
plt.axvline(upper_limit, color='green', linestyle='dashed', linewidth=2, label='+3 Std Dev')

# Calculate points at ±1 standard deviations
number_of_std_devs = 1
lower_limit = mu - number_of_std_devs * std
upper_limit = mu + number_of_std_devs * std

# Draw vertical lines at ±3 standard deviations
plt.axvline(lower_limit, color='blue', linestyle='dashed', linewidth=2, label='-3 Std Dev')
plt.axvline(upper_limit, color='blue', linestyle='dashed', linewidth=2, label='+3 Std Dev')


# Customize the plot
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency,Density')

# Save the plot as a PNG file
plt.savefig('normal_distribution_plot.png')

# Show the plot
plt.show()


plt.figure(figsize=(12, 8))

heatmap_data = df.pivot(index='year', columns='month', values='rain')
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=.5)

# Set labels and title
plt.xlabel('Month')
plt.ylabel('Year')
plt.title('Rainfall Heatmap')

# Save the plot as a PNG file
plt.savefig('heatmap.png')
# Show the plot
plt.show()

# Extract relevant columns (temperature and rainfall)
temperature_column = df['meant']  # You can use 'maxtp', 'mintp', or other temperature columns
rainfall_column = df['rain']

# Create masked arrays to handle NaN values
temperature_masked = ma.masked_invalid(temperature_column)
rainfall_masked = ma.masked_invalid(rainfall_column)

# Calculate the covariance between temperature and rainfall
covariance_temperature_rainfall = np.ma.cov(temperature_masked, rainfall_masked, allow_masked=True)[0, 1]

# Print the result
print(f'Covariance between temperature and rainfall: {covariance_temperature_rainfall}')


# Calculate the correlation coefficient
correlation_coefficient = np.ma.corrcoef(temperature_masked, rainfall_masked, allow_masked=True)[0, 1]


# Print the result
print(f'Correlation coefficient between temperature and rainfall: {correlation_coefficient}')


plt.figure(figsize=(12, 8))


# Extract 'rain' column for clustering
X = df[['rain']] # In Pandas, double brackets [['rain']] are used to create a DataFrame with a single column. The double brackets are used to create a DataFrame with a single column, rather than extracting the column as a Pandas Series. The reason for using double brackets in this context is that certain operations and methods in scikit-learn or other libraries might expect input features to be in the form of a DataFrame, even if there's only one column. Using double brackets ensures that you maintain a DataFrame structure. For example, when using scikit-learn for machine learning, the fit method typically expects the input features to be a 2D array-like object.
X = X.dropna(subset=['rain']) # Exclude NaN from Clustering
df = df.loc[X.index]
# Create a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model and predict the clusters
df['cluster'] = kmeans.fit_predict(X)

# Scatter plot to visualize the clusters
sns.scatterplot(x='rain', y='cluster', data=df, palette='viridis', marker='o')

# Display the centroids
plt.scatter(kmeans.cluster_centers_, range(3), color='red', marker='X', s=200, label='Centroids')

# Customize the plot
plt.title('K-means Clustering of Rain Data')
plt.xlabel('Rainfall')
plt.ylabel('Cluster')
plt.legend()

# Show the plot
plt.show()

# Clean the data, replace non mumber values with the mean
df = df.fillna(df.mean())

# Define predictor variables (X) and the target variable (y)
X = df[['meant', 'maxtp', 'mintp', 'mnmax', 'mnmin', 'gmin', 'wdsp', 'maxgt', 'sun']]
y = df['rain']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model coefficients and evaluation metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Coefficients: [ 5.18177332e+01 -2.04377762e+00  1.38940625e+00 -3.43689401e+01
#  -1.37278847e+01  2.32015596e-01  4.81756830e+00  1.27576005e+00
#  -1.99899128e-02]
# Intercept: 80.14481059148908
# Mean Squared Error: 1744.856837479092
# R-squared: 0.29766704701551816


# Coefficients:

# Each coefficient represents the change in the target variable (dependent variable) for a one-unit change in the corresponding feature (independent variable), while keeping other features constant.
# For example, a coefficient of 51.8177 for the first feature means that, on average, a one-unit increase in that feature is associated with an increase of approximately 51.82 units in the target variable.
# Intercept:

# The intercept represents the predicted value of the target variable when all independent variables are zero.
# In this case, when all features are zero, the predicted value of the target variable is approximately 80.14.
# Mean Squared Error (MSE):

# The MSE is a measure of how well the model's predictions match the actual values.
# Lower MSE values indicate better model performance, and it represents the average squared difference between the predicted values and the actual values.
# R-squared (R²):

# R-squared is a measure of how well the independent variables explain the variability in the dependent variable.
# It ranges from 0 to 1, where 0 indicates that the model does not explain any variability, and 1 indicates perfect explanation.
# In this case, an R-squared of 0.30 (30%) suggests that the model explains approximately 30% of the variability in the target variable.




# Clean the data, replace non mumber values with the mean
df = df.fillna(df.mean())


# Assuming 'df' is your DataFrame with weather data
# Create a binary target variable 'rainy' based on a threshold for 'rain'
df['rainy'] = (df['rain'] > 50).astype(int)

# Select features and target variable
features = ['meant', 'maxtp', 'mintp', 'mnmax', 'mnmin', 'gmin', 'wdsp', 'maxgt', 'sun']
X = df[features]
y = df['rainy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
#print(df)
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_report_str)

# Accuracy: 0.8393
# Confusion Matrix:
# [[ 11   8]
#  [ 19 130]]
# Classification Report:
#               precision    recall  f1-score   support

#            0       0.37      0.58      0.45        19
#            1       0.94      0.87      0.91       149

#     accuracy                           0.84       168
#    macro avg       0.65      0.73      0.68       168
# weighted avg       0.88      0.84      0.85       168

# The accuracy is the ratio of correctly predicted instances to the total instances.
# In this case, the classifier achieved an accuracy of approximately 83.93%. 
# It suggests that the model is correct in its predictions about 83.93% of the time.

# - Precision: For class 0 (non-rainy), precision is 0.37, indicating that among the instances predicted as non-rainy, only 37% were actually non-rainy. For class 1 (rainy), precision is 0.94, suggesting that among the instances predicted as rainy, 94% were actually rainy.
# - Recall: For class 0, recall is 0.58, indicating that 58% of actual non-rainy instances were correctly predicted. For class 1, recall is 0.87, meaning that 87% of actual rainy instances were correctly predicted.
# - F1-score: The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall.
# - Support: The number of instances for each class in the test set.

# The model has good overall accuracy, but there is room for improvement, especially in precision and recall for class 0 (non-rainy). Depending on your specific use case, you might want to further optimize the model or consider different evaluation metrics.
