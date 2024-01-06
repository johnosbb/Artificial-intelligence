import pandas as pd
import numpy as np
import matplotlib.pyplot as qplot
from scipy.stats import probplot

# Specify the values to be treated as missing
missing_values = ["", "NA", "N/A", "NaN"]


file_path = './data/weather_data.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path, na_values=missing_values)
# Convert the 'rain' column to numeric, handling non-numeric values
# df['rain'] = pd.to_numeric(df['rain'], errors='coerce')
# Convert blank spaces to numeric for all columns
df = df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

# Extract the 'rain' column
rain_data = df['rain']

# Create a Q-Q plot for the 'rain' data
probplot(rain_data, dist="norm", plot=qplot)
qplot.title("Q-Q Plot for Rain Data")
qplot.xlabel("Theoretical Quantiles")
qplot.ylabel("Sample Quantiles")
qplot.grid(True)

# Show the plot
qplot.show()
