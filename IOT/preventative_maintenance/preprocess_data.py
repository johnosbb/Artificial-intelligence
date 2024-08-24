import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot,norm
import seaborn as sns
import numpy.ma as ma
import os


# Specify the values to be treated as missing
missing_values = ["", "NA", "N/A", "NaN"]

current_directory = os.getcwd()
print("Current Directory:", current_directory)

dir_path = './preventative_maintenance/data/' # identify the directory path that holds the data


                 
# define column names for easy indexing
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names

# read data
train = pd.read_csv((dir_path+'train_FD001.txt'), sep='\s+', header=None, names=col_names)
test = pd.read_csv((dir_path+'test_FD001.txt'), sep='\s+', header=None, names=col_names)
y_test = pd.read_csv((dir_path+'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])


# Create a DataFrame with row numbers from 1 to 100
row_numbers = pd.DataFrame({'unit_nr': range(1, 101)})

# Add the row numbers as a new column before the current column
y_test = pd.concat([row_numbers, y_test], axis=1)

print(f"y_test = \n {y_test}")

# Iterate through y_test DataFrame
for index, row in y_test.iterrows():
    # Extract the unit number and corresponding RUL value
    unit_nr = row['unit_nr']
    rul = row['RUL'] # this is the remain uselful life value
    # Find rows in test DataFrame where "unit_nr" matches
    test_rows_with_unit_nr = test[test['unit_nr'] == unit_nr]
    # Set the "RUL" column for those rows to the corresponding RUL value from y_test
    test.loc[test_rows_with_unit_nr.index, 'RUL'] = rul



print("Loaded data sets\n")
print(f"train = \n {train}")
print(f"test = \n {test}")
print(f"y_test = \n {y_test}")