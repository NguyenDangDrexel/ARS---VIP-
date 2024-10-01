import numpy as np
import matplotlib.pyplot as plt 
import math 
""" 
This program is done 
This program can: 
    Solve the terminal_velocity at steady-state vs altitude for 4 cases 
        We can change the case by changing parameter case 
    Based on this terminal velocity to plot Re vs altitude 
        Can change this function

Does not affect by the m_virtual as acceleration = 0 
"""
import sys

# In order to import the file in the parent directory, we add the directory containing the parent file to the sys.path 
# setting path
path = r"C:\New folder\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project"
sys.path.append(path) 
from Models.ThermodynamicModels import Balloon
from Models.Atmospheric_models.AtmosphericModel import Pressure, Temperature, Density


# As I use the data of this file for drawing the graph in DynamicModel. Therefore, the plotting mode is set to be False by default  

m = 7 # total mass of the balloon: helium + latex + payload  (kg) 
Cd = 0.44 # constant drag coefficient 
g = 9.8 # gravitational accerleration (m/s^2) 
initial_params = [101325,590,293, 0] # [P0,n0,T0,delta_P]

"""
    Calculate the state of a constant volume balloon at a given altitude.

    Parameters:
    h (float): Altitude in meters
    initial_params (list): Initial parameters in the following order:
        - P0 (float): Initial pressure (Pa)
        - n0 (float): Initial number of moles (mol)
        - T0 (float): Initial temperature (K)
        - delta_P (float): Pressure difference (Pa)

    Returns:
    dict: Dictionary containing the balloon's state:
        - 'Pressure': P (Pa)
        - 'Volume': V (m^3)
        - 'Temperature': T (K)
        - 'Area': A (m^2)
        - 'Radius': r (m)
        - 'Number of moles': n (mol)
    """
#Area of the balloon 
# This function describe the terminal velocity with altitude for constant volume, input: inital volume, h 
def terminal_velocity (h:float,case:int,initial_params:list, m=7 ) -> list: 
    """

    Parameters: 
        h (float): Altitude in meters

        initial_params (list): Initial parameters in the following order:
            - P0 (float): Initial pressure (Pa)
            - n0 (float): Initial number of moles (mol)
            - T0 (float): Initial temperature (K)
            - delta_P (float): Pressure difference (Pa) 
        total mass of the balloon: helium + latex + payload  (kg). Default value = 7 (kg)

        case (int): 
            1: constant volume balloon
            2: iostherma balloon
            3: adiabatic balooon 
            4: simultaneous temperature balloon 
    Returns 
        v,case 
            v: ascend rate 
            case: thermodynamic model that we are using 
    """
    if case == 1: 
        V =  Balloon.constant_volume_balloon (h,initial_params= initial_params)['Volume']
        A = Balloon.constant_volume_balloon (h,initial_params= initial_params)['Area']
    elif case == 2: 
        V =  Balloon.iosthermal_balloon (h,initial_params= initial_params)['Volume']
        A = Balloon.iosthermal_balloon (h,initial_params= initial_params)['Area']
    elif case == 3: 
        V =  Balloon.adiabatic_balloon (h,initial_params= initial_params)['Volume']
        A = Balloon.adiabatic_balloon (h,initial_params= initial_params)['Area']
    elif case  == 4: 
        V =  Balloon.simultaneous_temperature_balloon (h,initial_params= initial_params)['Volume']
        A = Balloon.simultaneous_temperature_balloon (h,initial_params= initial_params)['Area']

    if (Density(h)*V)/(m) - 1 >0: 
        v = ( ((2*m) / (Cd*A*Density (h))) * g *( (Density(h)*V)/(m) - 1 ) ) ** (1/2)
    else: 
        v = -( ((2*m) / (Cd*A*Density (h))) * g *( 1 - (Density(h)*V)/(m)) ) ** (1/2)
        
    return v, case 

# Create linear space altitude 
start= 0 
stop = 10000
step = 1
altitude = np.arange (start,stop,step)
terminal_velocity_data = [] 
Volume = [] 
for h in altitude: 
    terminal_velocity_data.append (terminal_velocity (h= h, case = 1, initial_params= initial_params)[0])
    case = (terminal_velocity (h= h, case = 1, initial_params= initial_params)[1])

# Plotting 

plt.scatter (altitude, terminal_velocity_data)
if case == 1: 
    plt.title ('Steady State Solution of velocity vs alitude - constant volumme balloon')
elif case == 2: 
    plt.title ('Steady State Solution of velocity vs alitude - isothermal balloon')
elif case == 3: 
    plt.title ('Steady State Solution of velocity vs alitude - adiabatic  balloon')
elif case == 4: 
    plt.title ('Steady State Solution of velocity vs alitude - simultaneous temperature balloon')
plt.xlabel ('altitude (m)')
plt.ylabel ('ascend rate (m/s)')
plt.grid (True)
plt.show ()

