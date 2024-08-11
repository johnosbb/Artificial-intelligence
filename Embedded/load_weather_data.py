import csv
import matplotlib.pyplot as plt

# Define the CSV filename
csv_filename = './data/aggregated_weather_data.csv'

# Initialize empty lists to store temperature and humidity values
t_list = []
h_list = []
dates = []

# Open the CSV file and read the data
with open(csv_filename, 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        try:
            # Extract and clean temperature and humidity values
            temp = float(row['average_temperature'].replace(' °C', '').strip())
            humidity = float(row['humidity'].replace('%', '').strip())
            date = row['date']  # Extract the date
            
            # Append values to the lists
            t_list.append(temp)
            h_list.append(humidity)
            dates.append(date)
        except ValueError as e:
            print(f"Error processing row {row}: {e}")

# Define the interval for date ticks
tick_interval = max(1, len(dates) // 10)  # Show at most 10 date labels

# Create the plot
plt.figure(figsize=(12, 6))

# Plot temperature
plt.subplot(2, 1, 1)
plt.plot(dates, t_list, label='Temperature (°C)', color='tab:red')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Over Time')
plt.xticks(ticks=dates[::tick_interval], rotation=45)
plt.legend()

# Plot humidity
plt.subplot(2, 1, 2)
plt.plot(dates, h_list, label='Humidity (%)', color='tab:blue')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Humidity Over Time')
plt.xticks(ticks=dates[::tick_interval], rotation=45)
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
