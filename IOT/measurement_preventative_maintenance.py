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


file_path = './data/predictive_maintenance.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, na_values=missing_values)
                 
                 
# Display the DataFrame
print(df)




# Calculate the variance of the 'process_temperature_k' column
# The variance is a squared measure of how much each individual process_temperature_k value deviates from the mean.
# The larger the variance, the more spread out the data points are from the mean. 
# The unit of variance is squared, so it doesn't have the same unit as the original data.
# If you want a measure in the same unit as your original data, you can take the square root of the variance to get the standard deviation.
df['process_temperature_k_variance'] = df['process_temperature_k'].var()





# Calculate mean and Z-score
mean_process_temperature_k = df['process_temperature_k'].mean()
std_dev_process_temperature_k = df['process_temperature_k'].std()

df['mean_process_temperature_k'] = mean_process_temperature_k
df['z-score'] = (df['process_temperature_k'] - mean_process_temperature_k) / std_dev_process_temperature_k

min_process_temperature_k = df['process_temperature_k'].min()
max_process_temperature_k = df['process_temperature_k'].max()


# Specify the window size for the rolling analysis (e.g., 3 days)
window_size = 3

# Perform a rolling mean calculation
df['process_temperature_k_rolling_mean'] = df['process_temperature_k'].rolling(window=window_size).mean()


# Specify the overlap size for a hopping window
overlap = 2



# Adjust the index to create overlapping windows
df['process_temperature_k_hopping_mean'] = df['process_temperature_k_rolling_mean'].shift(-overlap)


# Display the DataFrame
print(df)



# Plotting the bar chart
df.plot(x='udi', y='process_temperature_k', kind='bar', figsize=(10, 6), legend=False)

# Adjusting x-axis labels to every nth item (e.g., every 2nd item)
n = 1000
plt.xticks(range(0, len(df), n), df['udi'].iloc[::n], rotation=45) # df['year'].iloc[::n]: This selects every nth year from the 'year' column of the DataFrame.

# Adding labels and title
plt.xlabel('UDI')
plt.ylabel('Process Temperature K')
plt.title('Process Temperature K Data Over Time')


# Save the plot as a PNG file
plt.savefig('./markdown/images/process_temperature_k_plot.png')

# Display the plot
plt.show()


## Scatter Plots

plt.figure(figsize=(10, 6))

plt.scatter(df['process_temperature_k'], df['air_temperature_k'])
plt.xlabel('process_temperature_k')
plt.ylabel('air_temperature_k')
plt.title('Scatter Plot: Process Temperature vs. Air Temperature')

# Save the plot as a PNG file
plt.savefig('process_temperature_air_temperature_scatter_plot.png')
# Display the plot
plt.show()