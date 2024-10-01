import pandas as pd 
from statistics import median as med 
import sys
from datetime import datetime
import numpy as np 
import plotly.express as px 
import plotly.graph_objects as go

# Assuming the DataFrame is already loaded and time is converted to seconds as before
path = r"C:\Drexel\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project\Data\20231014.xlsx"
df = pd.read_excel(path, skiprows=23)

# Convert time to seconds (same as before)
initial = pd.to_timedelta(df[' time [h:m:s GMT]'].iloc[0])
time_converted_lx = [(pd.to_timedelta(df[' time [h:m:s GMT]'].iloc[i]) - initial).total_seconds() for i in range(len(df[' time [h:m:s GMT]']))]
df[' time [h:m:s GMT]'] = time_converted_lx

# Define start and end indices
first_index = list(df[' time [h:m:s GMT]']).index(0)
last_index = list(df[' time [h:m:s GMT]']).index(11136)

# Extract the necessary data
ascend_rate = df[' iMet ascent rate [m/s]'][first_index:last_index].tolist()
time = df[' time [h:m:s GMT]'][first_index:last_index].tolist()
altitude_PTU = df[' altitude (from iMet PTU) [km]'][first_index:last_index].tolist()

# Function to calculate mean of a list
def calculate_mean(data):
    return sum(data) / len(data)

# Function to calculate standard deviation of a list
def calculate_std(data, mean):
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

# Function to perform Z-score normalization
def z_score_normalization(data):
    mean = calculate_mean(data)
    std = calculate_std(data, mean)
    z_scores = [(x - mean) / std for x in data]
    return z_scores

# Function to replace extreme Z-scores
def replace_extreme_z_scores(data, z_scores, threshold=20):
    median = med(data)
    modified_data = []
    for i, z in enumerate(z_scores):
        if z > threshold or z < -threshold:
            modified_data.append(median)
        else:
            modified_data.append(data[i])
    return modified_data


# Chunk the ascend_rate into 25 subfolds
chunk_size = len(ascend_rate) // 25
cleaned_ascend_rate = []

for i in range(0, len(ascend_rate), chunk_size):
    chunk = ascend_rate[i:i + chunk_size]
    z_scores = z_score_normalization(chunk)
    cleaned_chunk = replace_extreme_z_scores(chunk, z_scores, threshold=3)
    cleaned_ascend_rate.extend(cleaned_chunk)

# Plot the cleaned ascend rate over time
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=cleaned_ascend_rate, mode='markers', name='Cleaned Ascend Rate (m/s)'))
fig.update_layout(title_text='Cleaned Ascend Rate Over Time', xaxis_title='Time (s)', yaxis_title='Ascend Rate (m/s)')
fig.show()

# Plot the altitude over time
fig = px.line(x=time, y=altitude_PTU, labels={'x': 'Time (s)', 'y': 'Altitude (km)'}, title='Altitude from iMet Over Time')
fig.show()
