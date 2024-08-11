import requests
import json
import os
import csv
from datetime import datetime, timedelta

# The base URL for the historical weather API
base_url = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx'

def load_token_references(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def get_token_reference(token_name, references):
    return references.get(token_name)

def get_weather_data_for_month(year, month, api_key):
    start_date = f'{year}-{month:02d}-01'
    end_date = (datetime(year, month, 1) + timedelta(days=31)).replace(day=1) - timedelta(days=1)
    end_date = end_date.strftime('%Y-%m-%d')
    
    params = {
        'key': api_key,
        'q': 'London',  # Change this to the location you want
        'format': 'json',
        'date': start_date,
        'enddate': end_date,
        'tp': '24'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve data for {start_date} to {end_date}")
        return None

def get_weather_data_for_year(year, api_key):
    all_weather_data = []

    for month in range(1, 13):
        monthly_data = get_weather_data_for_month(year, month, api_key)
        if monthly_data and 'weather' in monthly_data['data']:
            all_weather_data.extend(monthly_data['data']['weather'])
    
    return all_weather_data

def get_weather_data_for_years(start_year, end_year, api_key):
    all_weather_data = []

    for year in range(start_year, end_year + 1):
        print(f"Retrieving data for year {year}...")
        yearly_data = get_weather_data_for_year(year, api_key)
        if yearly_data:
            all_weather_data.extend(yearly_data)
    
    return all_weather_data

# Set up your current directory and load API token
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)
token_references_file = './config/tokens.json'
token_name = 'weather_online'  
token_references = load_token_references(token_references_file)
api_key = get_token_reference(token_name, token_references)
print(f"API Key: {api_key}")

# Define the range of years
start_year = 2010
end_year = 2022

# Get weather data for the entire range of years
all_weather_data = get_weather_data_for_years(start_year, end_year, api_key)

# Save the aggregate data to a JSON file
output_json_filename = './data/aggregate_weather_data.json'
with open(output_json_filename, 'w', encoding='utf-8') as json_file:
    json.dump(all_weather_data, json_file, indent=4)

# Save the aggregated weather data to a CSV file
csv_filename = './data/aggregated_weather_data.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['date', 'average_temperature', 'humidity']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    for day in all_weather_data:
        date = day['date']
        avg_temp = day['avgtempC']
        humidity = day['hourly'][0]['humidity']
        writer.writerow({
            'date': date,
            'average_temperature': f"{avg_temp} °C",
            'humidity': f"{humidity}%"
        })
        print(f"Date: {date}, Avg Temp: {avg_temp}°C, Humidity: {humidity}%")

print(f"Aggregated weather data has been saved to {csv_filename}")
