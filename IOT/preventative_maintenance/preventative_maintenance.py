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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error, r2_score,precision_score
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, Dropout
from scipy.stats import shapiro, jarque_bera,anderson,kstest

# Data Source: https://github.com/mapr-demos/predictive-maintenance/tree/master/notebooks/jupyter/Dataset/CMAPSSData

SHOW_NOISE_ANALYSIS=False
SHOW_NOISE_ANALYSIS_WITH_ROLLING_AVERAGE=False
SHOW_NOISE_ANALYSIS_COMPARATIVE=False
SHOW_MAIN_STATISTICS_FOR_DATASET=True
SHOW_MAIN_STATISTICS_FOR_S2=False
SHOW_S2_NOISE_ANALYSIS=False
TEST_FOR_NORMALITY=True
SHOW_NOISE_ANALYSIS_S2_WITH_ROLLING_AVERAGE=False
SHOW_FREQUENCY_DISTRIBUTION_CHART=True

# Compute linear declining RUL is computed 
def add_remaining_useful_life(df):
    
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame
  



# Specify the values to be treated as missing
missing_values = ["", "NA", "N/A", "NaN"]


dir_path = './data/' # identify the directory path that holds the data


                 
# define column names for easy indexing
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names

# read data
train = pd.read_csv((dir_path+'train_FD001.txt'), sep='\s+', header=None, names=col_names)
test = pd.read_csv((dir_path+'test_FD001.txt'), sep='\s+', header=None, names=col_names)
y_test = pd.read_csv((dir_path+'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])
test["RUL"] = y_test # This line adds a new column named 'RUL' to the test DataFrame and assigns the values from the y_test DataFrame. It essentially adds the RUL information to the test data, allowing you to have a complete DataFrame with both sensor readings and the corresponding RUL values.

#Inspect the data for sensor number two
print(train["s_2"])

if SHOW_MAIN_STATISTICS_FOR_S2:       
    print(train["s_2"].describe().transpose())

if SHOW_MAIN_STATISTICS_FOR_DATASET:       
    print(train.describe().transpose())
    
if TEST_FOR_NORMALITY:
    test_statistic, p_value = shapiro(train["s_2"]) 
    print("Shapiro-Wilk Test Statistic:", test_statistic)
    print("Shapiro-Wilk P-value:", p_value)  
    
    _, p_value = jarque_bera(train["s_2"]) 
    print("Jarque-Bera P-value:", p_value)
    
    result = anderson(train["s_2"])
    print(f"Anderson Test for Normality: {result}")
    
    _, p_value = kstest(train["s_2"], 'norm')
    print(f"Kolmogorov-Smirnov Test for Normality: {p_value}")   
# We drop any columns that are of no use to us such as columns where the data does not change.
columns_to_drop = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19', 'setting_3']
train = train.drop(columns=columns_to_drop)
test= test.drop(columns=columns_to_drop)

train = add_remaining_useful_life(train)
test = add_remaining_useful_life(test)
# Add a new column "needs_maintenance" based on the values in the "RUL" column
train['needs_maintenance'] = (train['RUL'] <= 10).astype(int)
test['needs_maintenance'] = (test['RUL'] <= 10).astype(int)

print(train[index_names+['RUL']+['needs_maintenance']].head())

if SHOW_S2_NOISE_ANALYSIS:
    values = train['s_2'] 
    i = 1
    plt.figure(figsize=(10, 20))
    plt.plot(values) # plot the actual data
    plt.title("S2", y=0.5, loc='right')
    i += 1
    plt.show()
    

if SHOW_NOISE_ANALYSIS:
    values = train[train.time_cycles > 1] 
    groups = ['s_2','s_3','s_4','s_6','s_7','s_8','s_9','s_10']
    i = 1
    plt.figure(figsize=(10, 20))
    for group in groups:
        plt.subplot(len(groups), 1, i) # The subplot function is used to arrange multiple plots in a grid. len(groups): The total number of rows in the subplot grid. In our case, it's the length of the groups list, so each sensor will have its own row.
        plt.plot(values[group]) # plot the actual data
        plt.title(group, y=0.5, loc='right')
        i += 1
    plt.show()
    
    
train['rolling_mean'] = train.groupby('unit_nr')['time_cycles'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_2'] = train.groupby('unit_nr')['s_2'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_3'] = train.groupby('unit_nr')['s_3'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_4'] = train.groupby('unit_nr')['s_4'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_6'] = train.groupby('unit_nr')['s_6'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_7'] = train.groupby('unit_nr')['s_7'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_8'] = train.groupby('unit_nr')['s_8'].rolling(window=8, min_periods=1).mean().reset_index(level=0, drop=True)                

if SHOW_NOISE_ANALYSIS_WITH_ROLLING_AVERAGE:
    values = train[train.time_cycles > 1] 
    groups = ['rolling_average_s_1','rolling_average_s_2','rolling_average_s_3','rolling_average_s_4','rolling_average_s_5','rolling_average_s_6']
    i = 1
    plt.figure(figsize=(10, 20))
    for group in groups:
        plt.subplot(len(groups), 1, i) # The subplot function is used to arrange multiple plots in a grid. len(groups): The total number of rows in the subplot grid. In our case, it's the length of the groups list, so each sensor will have its own row.
        plt.plot(values[group]) # plot the actual data
        plt.title(group, y=0.5, loc='right')
        i += 1
    plt.show()

if SHOW_NOISE_ANALYSIS_S2_WITH_ROLLING_AVERAGE:
    values = train['rolling_average_s_2'] 
    i = 1
    plt.figure(figsize=(10, 20))
    plt.plot(values) # plot the actual data
    plt.title("S2 Rolling Average", y=0.5, loc='right')
    i += 1
    plt.show()
    
    
if SHOW_NOISE_ANALYSIS_COMPARATIVE:
    values = train[train.time_cycles > 1] 
    groups = ['s_2','rolling_average_s_2']
    i = 1
    plt.figure(figsize=(10, 20))
    for group in groups:
        plt.subplot(len(groups), 1, i) # The subplot function is used to arrange multiple plots in a grid. len(groups): The total number of rows in the subplot grid. In our case, it's the length of the groups list, so each sensor will have its own row.
        plt.plot(values[group]) # plot the actual data
        plt.title(group, y=0.5, loc='right')
        i += 1
    plt.show()
    
    
X_train = train[['rolling_average_s_2','rolling_average_s_3','rolling_average_s_4','rolling_average_s_7','rolling_average_s_8']]
y_train = train[['RUL']]
X_test = [['rolling_average_s_2','rolling_average_s_3','rolling_average_s_4','rolling_average_s_7','rolling_average_s_8']]
y_test = test[['RUL']]

maintenance_instances_train = train.loc[train['needs_maintenance'] == 1]
columns_to_print = ['unit_nr', 'time_cycles', 'RUL', 'needs_maintenance']
#print(maintenance_instances_train[columns_to_print])

if SHOW_FREQUENCY_DISTRIBUTION_CHART:    
    # Extract the 's_2_data' column
    s_2_data = train['s_2']
    # Plot a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(s_2_data, bins=20, kde=False, color='blue', stat='density', element='step')

    # Fit a normal distribution to the data
    mu, std = norm.fit(s_2_data)
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
    plt.xlabel('S2 Sensor Data')
    plt.ylabel('Frequency,Density')

    # Save the plot as a PNG file
    plt.savefig('./markdown/images/FD001_normal_distribution_plot.png')
    # Show the plot
    plt.show()