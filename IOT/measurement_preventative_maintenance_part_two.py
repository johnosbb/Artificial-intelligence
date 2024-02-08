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

# Data Source: https://github.com/mapr-demos/predictive-maintenance/tree/master/notebooks/jupyter/Dataset/CMAPSSData

SHOW_NOISE_ANALYSIS=False
SHOW_NOISE_ANALYSIS_WITH_ROLLING_AVERAGE=False
SHOW_NOISE_ANALYSIS_COMPARATIVE=False


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
print(maintenance_instances_train[columns_to_print])