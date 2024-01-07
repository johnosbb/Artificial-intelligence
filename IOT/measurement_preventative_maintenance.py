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


SHOW_HISTOGRAM=False
SHOW_SCATTER=False
SHOW_BUBBLE_CHART=False
SHOW_Q_CHART=False
SHOW_FREQUENCY_DISTRIBUTION_CHART=False
SHOW_HEATMAP=True

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


if SHOW_HISTOGRAM:
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

if SHOW_SCATTER:
    
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


if SHOW_BUBBLE_CHART:
    # Bubble Chart with Every nth Value
    n = 100  # Change this value to skip every n-th data point
    plt.scatter(df['rotational_speed _rpm'][::n], df['torque_Nm'][::n], s=df['tool_wear_min'][::n], alpha=0.5)


    # Adding labels and title
    plt.xlabel('Rotational Speed RPM')
    plt.ylabel('Torque Nm')
    plt.title('Bubble Chart for Rotational Speed, Torque and Tool Wear')

    # Save the plot as a PNG file
    plt.savefig('./markdown/images/bubble_chart_tool_wear.png')

    # Display the plot
    plt.show()

if SHOW_Q_CHART:
    plt.figure(figsize=(10, 6))

    # Extract the 'tool_wear_min' column
    tool_wear_min = df['tool_wear_min']

    # Remove non-finite values (NaN and infinite)
    tool_wear_min = tool_wear_min[np.isfinite(tool_wear_min)]


    # Create a Q-Q plot for the 'tool_wear_min' data
    probplot(tool_wear_min, dist="norm", plot=plt)
    plt.title("Q-Q Plot for Tool Wear - Test for Normal Distribution")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig('./markdown/images/q_q_plot.png')

    # Show the plot
    plt.show()
    

if SHOW_FREQUENCY_DISTRIBUTION_CHART:    
    # Extract the 'tool_wear_min' column
    tool_wear_min = df['tool_wear_min']

    # Remove non-finite values (NaN and infinite)
    tool_wear_min = tool_wear_min[np.isfinite(tool_wear_min)]
    # Plot a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(tool_wear_min, bins=20, kde=False, color='blue', stat='density', element='step')

    # Fit a normal distribution to the data
    mu, std = norm.fit(tool_wear_min)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
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
    plt.xlabel('Tool Wear (mm)')
    plt.ylabel('Frequency,Density')

    # Save the plot as a PNG file
    plt.savefig('./markdown/images/normal_distribution_plot.png')
    # Show the plot
    plt.show()
    
    
if SHOW_HEATMAP: 
    # Select relevant numerical features for correlation analysis
    numerical_features = ['air_temperature_k', 'process_temperature_k', 'rotational_speed _rpm', 'torque_Nm', 'tool_wear_min']
    # Create a correlation matrix
    correlation_matrix = df[numerical_features].corr()
    # Plot the heatmap
    # plt.figure(figsize=(12, 14))
    heatmap=sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    # Rotate x-axis labels
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
    # Adjust layout to prevent label cropping
    plt.tight_layout()
    plt.show()
