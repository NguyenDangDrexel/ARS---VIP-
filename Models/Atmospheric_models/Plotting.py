# Plot the Temperature, Pressure vs altitude 
from AtmosphericModel import Pressure, Temperature , Density
import numpy as np 
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

start = 0 
stop = 35000
step = 100
Pressure_lx = []
Temperature_lx = [] 
Density_lx = [] 
altiude = np.arange (start = start, stop =stop ,step =step)
# Pressure_lx = 
for h in altiude: 
    Pressure_lx.append (Pressure (h))
    Temperature_lx.append (Temperature(h))
    Density_lx.append (Density(h))
# Create subplots
fig = make_subplots(rows=1, cols=3, subplot_titles=('Pressure vs Altitude', 'Temperature(K) vs Altitude', 'Density vs Altitude'))

# Add scatter plot traces for pressure
fig.add_trace(go.Scatter(x= altiude, y=Pressure_lx, mode='markers', name='Pressure', marker=dict(color='blue')), row=1, col=1)

# Add scatter plot traces for temperature
fig.add_trace(go.Scatter(x= altiude, y=Temperature_lx, mode='markers', name='Temperature', marker=dict(color='red')), row=1, col=2)

fig.add_trace(go.Scatter(x= altiude, y=Density_lx, mode='markers', name='Temperature', marker=dict(color='red')), row=1, col=3)

# Update layout
fig.update_xaxes(title_text='Altitude (m)', row=1, col=1)
fig.update_yaxes(title_text='Pressure (Pa)', row=1, col=1)
fig.update_xaxes(title_text='Altitude (m)', row=1, col=2)
fig.update_yaxes(title_text='Temperature (K)', row=1, col=2)
fig.update_xaxes(title_text='Altitude (m)', row=1, col=3)
fig.update_yaxes(title_text='Density (kg/m3)', row=1, col=3)

fig.update_layout(title='Pressure and Temperature vs Altitude')
fig.show()