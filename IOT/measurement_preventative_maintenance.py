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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error, r2_score

SHOW_ROLLING_MEAN=False
SHOW_HISTOGRAM=False
SHOW_SCATTER=False
SHOW_BUBBLE_CHART=False
SHOW_Q_CHART=False
SHOW_FREQUENCY_DISTRIBUTION_CHART=False
SHOW_HEATMAP=False
SHOW_COVARIANCE=False
SHOW_CLUSTER=False
SHOW_CORRELATION_WITH_LINEAR_REGRESSION=False
SHOW_PREDICTION_ACCURACY=False
SHOW_XGBOOST_CLASSIFICATION_ACCURACY=False


# Specify the values to be treated as missing
missing_values = ["", "NA", "N/A", "NaN"]


file_path = './data/predictive_maintenance.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, na_values=missing_values)
                 
                 
# Display the DataFrame
print(df)


if SHOW_ROLLING_MEAN:
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

if SHOW_COVARIANCE: 
    # Extract relevant columns (temperature and rainfall)
    rotational_speed = df['rotational_speed_rpm']  
    tool_wear_min = df['tool_wear_min']

    # Create masked arrays to handle NaN values
    rotational_speed_filtered = ma.masked_invalid(rotational_speed)
    tool_wear_min_filtered = ma.masked_invalid(tool_wear_min)

    # Calculate the covariance between tool_wear_min_filtered and rotational_speed_filtered
    covariance_rotational_speed_tool_wear = np.ma.cov(rotational_speed_filtered, tool_wear_min_filtered, allow_masked=True)[0, 1]

    # Print the result
    print(f'Covariance between rotational speed, tool wear: {covariance_rotational_speed_tool_wear}')

    # Calculate the correlation coefficient
    correlation_coefficient = np.ma.corrcoef(rotational_speed_filtered, tool_wear_min_filtered, allow_masked=True)[0, 1]

    # Print the result
    print(f'Correlation coefficient between rotational speed and tool wear,: {correlation_coefficient}')
    
    
        # Calculate the covariance between tool_wear and torque
    covariance_rotational_speed_tool_wear = np.ma.cov(df['torque_Nm'], tool_wear_min_filtered, allow_masked=True)[0, 1]

    # Print the result
    print(f'Covariance between torque, tool wear: {covariance_rotational_speed_tool_wear}')

    # Calculate the correlation coefficient
    correlation_coefficient = np.ma.corrcoef(df['torque_Nm'], tool_wear_min_filtered, allow_masked=True)[0, 1]

    # Print the result
    print(f'Correlation coefficient between torque and tool wear,: {correlation_coefficient}')
    
    

if SHOW_CLUSTER:
    plt.figure(figsize=(12, 8))

    # Extract 'torque_Nm' column for clustering
    X = df[['torque_Nm']] # In Pandas, double brackets [['torque_Nm']] are used to create a DataFrame with a single column. The double brackets are used to create a DataFrame with a single column, rather than extracting the column as a Pandas Series. The reason for using double brackets in this context is that certain operations and methods in scikit-learn or other libraries might expect input features to be in the form of a DataFrame, even if there's only one column. Using double brackets ensures that you maintain a DataFrame structure. For example, when using scikit-learn for machine learning, the fit method typically expects the input features to be a 2D array-like object.
    X = X.dropna(subset=['torque_Nm']) # Exclude NaN from Clustering if required
    df = df.loc[X.index]
    number_of_clusters = 10
    # Create a KMeans model with 3 clusters
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42)

    # Fit the model and predict the clusters
    df['cluster'] = kmeans.fit_predict(X)

    # Scatter plot to visualize the clusters
    sns.scatterplot(x='torque_Nm', y='cluster', data=df, palette='viridis', marker='o')

    # Display the centroids
    plt.scatter(kmeans.cluster_centers_, range(number_of_clusters), color='red', marker='X', s=200, label='Centroids')

    # Customize the plot
    plt.title('K-means Clustering of Torque')
    plt.xlabel('Torque')
    plt.ylabel('Cluster')
    plt.legend()

    # Show the plot
    plt.show()


if SHOW_CORRELATION_WITH_LINEAR_REGRESSION:
        

    # udi,product_id,product_type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_Nm,tool_wear_min,target,failure_product_type
    # Define predictor variables (X) and the target variable (y)
    X = df[[ 'air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_Nm' ]]
    y = df['tool_wear_min']


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

    # Each coefficient represents the change in the target variable (dependent variable) for a one-unit change in the 
    # corresponding feature (independent variable),
    # while keeping other features constant.
    # For example, a coefficient of 0.01940713 for the first feature means that, 
    # on average, a one-unit increase in that feature is associated with an increase of approximately 51.82 units in the target variable.
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



if SHOW_PREDICTION_ACCURACY: # uses a DecisionTreeClassifier

    # Select features and target variable
    # udi,product_id,product_type,air_temperature_k,process_temperature_k,rotational_speed_rpm,torque_Nm,tool_wear_min,target,failure_product_type
    features = ['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_Nm','tool_wear_min']
    X = df[features]
    y = df['target']

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

# Accuracy: 0.9795
# Confusion Matrix:
# [[1914   25]
#  [  16   45]]

# Confusion Matrix:

# True Positives (TP): 45 - The model correctly predicted 45 instances of failure.
# True Negatives (TN): 1914 - The model correctly predicted 1914 instances of non-failure.
# False Positives (FP): 25 - The model incorrectly predicted 25 instances as failure when they were not.
# False Negatives (FN): 16 - The model incorrectly predicted 16 instances as non-failure when they were.

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99      1939
#            1       0.64      0.74      0.69        61

#     accuracy                           0.98      2000
#    macro avg       0.82      0.86      0.84      2000
# weighted avg       0.98      0.98      0.98      2000

# Classification Report:

# Precision: Precision is the ratio of true positives to the sum of true positives and false positives.
# In our case, the precision for class 0 (non-failure) is 0.99, meaning that when the model predicts non-failure, it is correct about 99% of the time.
# The precision for class 1 (failure) is 0.64, meaning that when the model predicts failure, it is correct about 64% of the time.
# Recall (Sensitivity): Recall is the ratio of true positives to the sum of true positives and false negatives. 
# The recall for class 1 is 0.74, indicating that the model captures 74% of the actual failures.
# F1-score: The F1-score is the harmonic mean of precision and recall. 
# It provides a balance between precision and recall. The F1-score for class 1 is 0.69.
# Support: The number of actual occurrences of the class in the specified dataset.

# Macro and Weighted Averages:
#   Macro Avg: The average of precision, recall, and F1-score for both classes, without considering class imbalance.
#   In our case, the macro average F1-score is 0.84.
#   Weighted Avg: The average of precision, recall, and F1-score, weighted by the number of samples in each class.
#   This is useful when there is an imbalance in the number of samples between classes. In your case, the weighted average F1-score is 0.98.

# In summary, our model has a high overall accuracy, but it's essential to consider precision, recall, and F1-score, especially for the class representing failure, to understand the performance of the model in predicting failures accurately and avoiding false positives/negatives.

if SHOW_XGBOOST_CLASSIFICATION_ACCURACY:    # uses an XBGClassifier
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
    
    
print(df.describe().transpose())
columns_to_drop = ['product_id', 'udi', 'product_type', 'failure_product_type']
df = df.drop(columns = columns_to_drop)
print(df)
corr = df.corr().round(1) # corr() is a Pandas DataFrame method used to compute the pairwise correlation of columns, excluding NA/null values. The .round(1) method is then used to round the correlation values to one decimal place.
print(f"corr: {corr}")
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True # The term "triu" stands for "upper triangle," 
# In the heatmap, only the lower triangle (excluding the main diagonal) is usually shown, as the upper triangle is symmetrically the same.
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
correlation_heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5},
 annot=True)
plt.title('Correlation Matrix Heatmap')
# Rotate x-axis labels
correlation_heatmap.set_xticklabels(correlation_heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
correlation_heatmap.set_yticklabels(correlation_heatmap.get_yticklabels(), rotation=45, horizontalalignment='right')
# Adjust layout to prevent label cropping
plt.tight_layout(rect=(0, 0, 1, 1))
plt.show()