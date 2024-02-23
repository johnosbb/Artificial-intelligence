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
import os

# Data Source: https://github.com/mapr-demos/predictive-maintenance/tree/master/notebooks/jupyter/Dataset/CMAPSSData

SHOW_NOISE_ANALYSIS=False
SHOW_NOISE_ANALYSIS_WITH_ROLLING_AVERAGE=False
SHOW_NOISE_ANALYSIS_COMPARATIVE=False
SHOW_MAIN_STATISTICS_FOR_DATASET=False
SHOW_MAIN_STATISTICS_FOR_S2=False
SHOW_S2_NOISE_ANALYSIS=False
TEST_FOR_NORMALITY=True
SHOW_NOISE_ANALYSIS_S2_WITH_ROLLING_AVERAGE=False
SHOW_FREQUENCY_DISTRIBUTION_CHART=False
SCALE_SENSOR_DATA=True
SEQUENTIAL_MODEL=True

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
    
window_size=8    
train['rolling_mean'] = train.groupby('unit_nr')['time_cycles'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_2'] = train.groupby('unit_nr')['s_2'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_3'] = train.groupby('unit_nr')['s_3'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_4'] = train.groupby('unit_nr')['s_4'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_6'] = train.groupby('unit_nr')['s_6'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_7'] = train.groupby('unit_nr')['s_7'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
train['rolling_average_s_8'] = train.groupby('unit_nr')['s_8'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)                


test['rolling_mean'] = test.groupby('unit_nr')['time_cycles'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
test['rolling_average_s_2'] = test.groupby('unit_nr')['s_2'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
test['rolling_average_s_3'] = test.groupby('unit_nr')['s_3'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
test['rolling_average_s_4'] = test.groupby('unit_nr')['s_4'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
test['rolling_average_s_6'] = test.groupby('unit_nr')['s_6'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
test['rolling_average_s_7'] = test.groupby('unit_nr')['s_7'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
test['rolling_average_s_8'] = test.groupby('unit_nr')['s_8'].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)                


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
y_train = train['needs_maintenance']
X_test = test[['rolling_average_s_2','rolling_average_s_3','rolling_average_s_4','rolling_average_s_7','rolling_average_s_8']]
y_test = test['needs_maintenance']

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
    plt.savefig('./preventative_maintenance/markdown/images/FD001_normal_distribution_plot.png')
    # Show the plot
    plt.show()
    
if SCALE_SENSOR_DATA:
    print(X_train)
    print("Selection:\n")
    print(X_train.iloc[:, 0:5])
    scaler = MinMaxScaler(feature_range=(0, 1)) # The MinMaxScaler is a class from the sklearn.preprocessing module in scikit-learn. It is used to scale numerical features between a specified range. In this case, the range is set to (0, 1).
    # Fit the scaler on the training data and transform both training and testing data
    X_train.iloc[:, 0:5] = scaler.fit_transform(X_train.iloc[:, 0:5]) # selects all rows and columns from index 1 to 5 (inclusive) in the'
    X_test.iloc[:, 0:5] = scaler.transform(X_test.iloc[:, 0:5])
    dim = X_train.shape[1]
    print(X_train)
    
# Sequential Model
    
# In the context of deep learning, a Sequential model is a linear stack of layers in which you build a neural network layer by layer, starting from the input layer and progressing through hidden layers until the output layer. Each layer in the Sequential model has weights that correspond to the layer that follows it.
# The Sequential model is part of the Keras library, a high-level neural networks API that is integrated into TensorFlow. Keras provides a convenient way to define and build neural network models, especially for beginners and researchers.

if SEQUENTIAL_MODEL:
    dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(32, input_dim = dim))  # Input layer with 32 nodes
    model.add(LeakyReLU()) # Activation function for the first layer
    model.add(Dropout(0.25))

    # The LeakyReLU activation introduces a small negative slope to the standard ReLU activation, and Dropout helps prevent overfitting by randomly setting a fraction of input units to zero during training.
    # Add a hidden layer: In the context of a neural network architecture, the term "hidden layer" refers to any layer between the input layer and the output layer. It's called "hidden" because it is not directly observable from the network's inputs or outputs during training or prediction.
    model.add(Dense(32)) 
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    
    # Add an output layer:
    # A Dense layer with a single node.
    # The activation function used here is the sigmoid activation function.
    # In a binary classification problem, where the output should be a probability between 0 and 1,
    # a common choice for the activation function in the output layer is the sigmoid function.
    # The sigmoid function squashes the output values to the range [0, 1], making it suitable for binary classification.    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # The choice of the optimizer and loss function in a neural network depends on the specific problem we are trying to solve.
    # RMSprop (Root Mean Square Propagation) is an optimization algorithm commonly used for training neural networks.
    # It adapts the learning rates of each parameter based on the average of recent gradients.
    # RMSprop helps address some issues with other optimizers like AdaGrad, particularly in the case of deep neural networks.
    # It adjusts the learning rates for each parameter individually, 
    # which can be beneficial in handling different scales and sparsity in the data.
    model.compile(optimizer ='rmsprop', loss ='binary_crossentropy', metrics = ['accuracy'])
    model.fit(X_train, y_train, batch_size = 32, epochs = 5,
        verbose = 1, validation_data = (X_test, y_test))
    print(f"X_test = {X_test} ")
    y_pred_prob  = model.predict(X_test)
    print(f"y_pred_prob = {y_pred_prob}")
    # Convert probabilities to binary predictions using a threshold of 0.5
    y_pred = (y_pred_prob >= 0.5).astype(int)
    print(f"y_pred = {y_pred}")
    pre_score = precision_score(y_test,y_pred, average='micro')
    print("Neural Network:",pre_score)

    
# The data is fed from the input to each of the 32 input neurons. The neurons
# are connected through channels. The channel is assigned a numerical value known as
# weight. The inputs are multiplied by the corresponding weight and their sum is sent as
# input to the neurons in the hidden layer. Each of these neurons is associated with a
# numerical value called the bias, which is added to the input sum. This value is then passed
# to a threshold function called the activation function. The activation function determines if
# a neuron will get activated or not. In this sequential model we used Leaky ReLU as our activation function for our
# first 2 layers. ReLU or Rectified Linear Unit is a popular activation function because it
# solves the vanishing gradient problem. The vanishing gradient problem occurs when these gradients become extremely 
# small as they are propagated backward through the layers of the network.
# In this recipe, we used the Leaky ReLU. Leaky ReLU solves a problem that ReLU has where big gradients can
# cause the neuron to never fire.
# The activated neuron passes its data to the next layer over the channels.
# This method allows the data to be propagated through the network. This is called forward
# propagation. In the output layer, the neuron with the highest layer fires and determines the
# output.