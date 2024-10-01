"""
This program solving the dynamic equation by using odeint (). In this program, the following assumptions are made: 
1. Drag coefficient is constant; Cd = 0.44. 
2. Density, Radius change with altitdue 
3. Density of atmosphere and Helium = 1.293 and 0.166 (kg/m^3), respectively 
4. Total mass of balloon: 7 (kg)
5. Initial Radius = 15.05 (m^3) 

This model works well.  
"""
import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 
import math  

# In order to import the file in the parent directory, we add the directory containing the parent file to the sys.path 
# setting path


import sys
path = r"C:\Drexel\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project"
sys.path.append(path) 
from SteadyStateSolution import * # note that whenever you change case, you need to change case in V_terminal file 

from Models.ThermodynamicModels import Balloon
from Models.Atmospheric_models.AtmosphericModel import Pressure, Temperature, Density 

plot = True      
# case 1: constant volume, case 2: isothermal; case 3: adiabatic; case 4: t_in = t_out 
# Want to compare with the steady state solution ??? => when compare, make sure that they have the same parameters 
compare = False    
#Drag coefficient  
Cd = 0.44 
V0 = 15.05 # for noised_file 
m = 7
C = 0.5 # m_air = C * rho * V 
g = 9.8 
initial_params = [101325,629,293, 0]  # [P0,n0,T0,delta_P]


"""
INPUT: case, non-uniform state-space 
OUTPUT: altitude_data solved by odeint ()
"""
Volume = [] 
def numerical_solultion_altitude_data (case, time_space): 
    time_space = time_space 
    # IMPORTANT: the form of the function matter. we need to put the highest order on left side, and the rest on the right side 
    def function (y,time_space):
        x,xdot = y
        h = x 
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
        
        m_air = C*Density(x) * V 
        Volume.append (V)

        if xdot >0: 
            xddot = (-m*g + Density(x)*V *g)/ (m +m_air)  - 1/2 *Density(x) * xdot**2 * A*Cd /(m + m_air)  
        else: 
            xddot = (-m*g + Density(x)*V *g)/ (m +m_air)  + 1/2 *Density(x) * xdot**2 * A*Cd /(m + m_air) 
        
        return xdot,xddot 

    solution2 = odeint (function, y0= [0,0], t = time_space) 

    numerical_altitude_data = solution2[:,0]
    numerical_ascend_rate_data = solution2[:,1]

    return numerical_altitude_data, numerical_ascend_rate_data,Volume
#Take solutions from ODEint 
case = 4 

if case == 1: 
    time_points2 = np.arange (0,3000,1) # when h = max 
elif case == 2: 
    time_points2 = np.arange (0,2960,1) # when h = 35000 
elif case == 3: 
    time_points2 = np.arange (0,4500,1) # when t larger than this value, odeint fails 
elif case ==4: 
    time_points2 = np.arange (0,4500,1)
numerical_altitude_data, numerical_ascend_rate_data,V = numerical_solultion_altitude_data (case =case, time_space=time_points2)

venting_point = 1000
initial_terminal_velocity = numerical_ascend_rate_data [10] # = velocity when time = 10
print (f'Altitude when t = {venting_point} (s)         :{numerical_altitude_data[venting_point]}')
print (f'Ascend_rate when t = {venting_point} (s)      :{numerical_ascend_rate_data [venting_point]}')
if plot: 
# Create a figure and axes with 1 row and 2 columns
    if compare: 
        fig, axes = plt.subplots(1, 3, figsize=(8, 6))
    else: 
        fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    if case  == 1:
        fig.suptitle('numerical solution - constant volume balloon')
    elif case == 2:
        fig.suptitle('numerical solution - isothermal balloon')

    elif case == 3:
        fig.suptitle('numerical solution - adiabatic balloon')

    elif case == 4: 
        fig.suptitle('numerical solution - Tin_=T_out balloon')

    # First plot
    axes[0].plot(time_points2, numerical_altitude_data, color='blue')
    axes[0].set_title('Altitude - odeint')
    axes[0].set_xlabel('t(s)')
    axes[0].set_ylabel('h(m)')
    axes[0].grid(True)

        # Second plot
    axes[1].plot(time_points2, numerical_ascend_rate_data, color='blue')
    axes[1].set_title('Velocity - odeint')
    axes[1].set_xlabel('t(s)')
    axes[1].set_ylabel('v(m/s)')
    axes[1].grid(True)
    yticks = np.linspace (0,20,11).tolist()
    yticks.append (initial_terminal_velocity) 
    axes[1].set_yticks (yticks)

    




    if compare: 
    # third plot: comparing numerical solution vs steady-state solution 
        axes[2].plot(numerical_altitude_data, numerical_ascend_rate_data, color='blue')
        # axes[2].scatter(Altitude_data,terminal_velocity_data, color='red', s = 10)
        axes[2].set_title('numerical solution vs steady-state solution')
        axes[2].set_xlabel('h(m)')
        axes[2].set_ylabel('v(m/s)')
        axes[2].grid(True) 
    plt.tight_layout()
    plt.show()


print (initial_terminal_velocity)
