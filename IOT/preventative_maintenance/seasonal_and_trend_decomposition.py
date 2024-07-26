import pandas as pd
import sesd
import numpy as np

# df = pd.read_csv('./data/Beach_Water_Quality_-_Automated_Sensors.csv',
#  header=0)
# df = df[df['Beach Name'] == 'Rainbow Beach']
# df = df[df['Water Temperature'] > -100]
# df = df[df['Wave Period'] > -100]
# waveheight = df[['Wave Height']].to_numpy()

# outliers_indices = sesd.seasonal_esd(waveheight, hybrid=True, max_anomalies=2)


# for idx in outliers_indices:
#  print("Anomaly index: {}, anomaly value: {}"\
#  .format(idx, waveheight[idx]))

import pandas as pd
import sesd
import numpy as np

# Load the data
df = pd.read_csv('./data/Beach_Water_Quality_-_Automated_Sensors.csv', header=0)

# Filter the data
df = df[df['Beach Name'] == 'Rainbow Beach']
df = df[df['Water Temperature'] > -100]
df = df[df['Wave Period'] > -100]

# Extract Wave Height as a one-dimensional array
waveheight = df['Wave Height'].to_numpy()

# Detect outliers
outliers_indices = sesd.seasonal_esd(waveheight, hybrid=True, max_anomalies=2)

# Print the anomalies
for idx in outliers_indices:
    print(f"Anomaly index: {idx}, anomaly value: {waveheight[idx]}")


# The results provided by the seasonal_esd function indicate that there are anomalies (outliers) in your Wave Height data at specific indices. Here's how to interpret these results:

# Anomaly Index:

# The Anomaly index refers to the position of the anomaly within the waveheight array. For example, an Anomaly index: 2847 means that the 2848th element (since indexing starts at 0) in the waveheight array is considered an anomaly.
# Anomaly Value:

# The anomaly value is the actual value of the Wave Height at the given index. For example, anomaly value: 0.74 at index 2847 means that the Wave Height at that position is 0.74 units (presumably meters or feet, depending on the dataset's unit).

# Index 2847:

# At this index, the Wave Height is 0.74, which the algorithm has flagged as an anomaly. This means that this value significantly deviates from the expected pattern of the data at that point in time.
# Index 2522:

# At this index, the Wave Height is 0.602, which is also flagged as an anomaly. Similarly, this value deviates significantly from what is expected based on the surrounding data.


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(waveheight, label='Wave Height')
plt.scatter(outliers_indices, waveheight[outliers_indices], color='red', label='Anomalies')
plt.xlabel('Index')
plt.ylabel('Wave Height')
plt.title('Wave Height with Anomalies')
plt.legend()
plt.show()


# Contextual Factors:

# Consider any external factors that might explain these anomalies. For example, unusual weather conditions, equipment malfunctions, or significant events could cause deviations in wave height.
# Data Patterns:

# Review the overall pattern and trend of the data. Determine if these anomalies represent outliers due to natural variability or if they indicate an underlying issue.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load the data
df = pd.read_csv('./data/Beach_Water_Quality_-_Automated_Sensors.csv', header=0)

# Filter the data
df = df[df['Beach Name'] == 'Rainbow Beach']
df = df[df['Water Temperature'] > -100]
df = df[df['Wave Period'] > -100]

# Extract Wave Height as a one-dimensional array
waveheight = df['Wave Height'].to_numpy()

# Perform STL decomposition
stl = STL(waveheight, period=365)  # Assuming daily data with an annual seasonality
result = stl.fit()

# Extract the components
seasonal = result.seasonal
trend = result.trend
resid = result.resid

# Plot the seasonal component
plt.figure(figsize=(10, 6))
plt.plot(seasonal, label='Seasonal')
plt.xlabel('Index')
plt.ylabel('Wave Height')
plt.title('Seasonal Component of Wave Height')
plt.legend()
plt.show()

# Plot the trend component
plt.figure(figsize=(10, 6))
plt.plot(trend, label='Trend')
plt.xlabel('Index')
plt.ylabel('Wave Height')
plt.title('Trend Component of Wave Height')
plt.legend()
plt.show()


# Plot the residual component
plt.figure(figsize=(10, 6))
plt.plot(resid, label='Residual')
plt.xlabel('Index')
plt.ylabel('Wave Height')
plt.title('Residual Component of Wave Height')
plt.legend()
plt.show()

# After STL Decomposition:

# Trend Component: Represents the underlying trend in the data, showing the long-term progression without seasonal effects.
# Seasonal Component: Captures the repetitive seasonal pattern in the data, showing variations that occur at regular intervals (e.g., monthly, yearly).
# Residual Component: Represents the remaining noise or irregular variations in the data after removing the trend and seasonal components.

# STL decomposes a time series by iteratively separating out the seasonal and trend components using LOESS (Locally Estimated Scatterplot Smoothing) smoothing.
#  This allows for a clear identification of the underlying trends and periodic patterns in the data, 
# which can be analyzed and interpreted separately. By visualizing these components, 
# you can better understand the structure and behavior of the time series data.