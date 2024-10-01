import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 
import math 
#In this file is good, should not change anything. DONE 
# In this model: atmosphere is the constant => it's only valid for small range of altitude: 0 - 500 (m) => it's only valid to predict the initial terminal velocity 

# Compare with https://predict.sondehub.org/, the predicted terminal velocity is similar for the range 4.5 - 10.
# The dynamic equation is similar to predicted website in the range: 4.5 - 10 (m/s) and approximately similar for all the values of mass 
# => What happen with the small teminal velocity ? 

import sys

# In order to import the file in the parent directory, we add the directory containing the parent file to the sys.path 
# setting path
path = r"C:\New folder\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project"
sys.path.append(path) 
from Models.ThermodynamicModels import Balloon
from Models.Atmospheric_models.AtmosphericModel import Pressure, Temperature, Density 

simulation_time = 5
V0 = 15.05  # Volume of gas inside the balloon (m^3)
m = 7
g = 9.8 
Cd = 0.44
initial_params = [101325,626,293, 0]
h= 0 
beta = 1/2 * Cd *Density(0) *Balloon.constant_volume_balloon (h,initial_params)['Area']/ m 
alpha = (Density(0) *Balloon.constant_volume_balloon (h,initial_params)['Volume']*g - m*g) / m # This is initial acceleration: 15.7 m/s^2 => it will reach terminal velocity at 0.5 s 
v0 = (alpha/beta) ** (1/2)
def v (t,alpha,beta): 
    v = v0 * ( (1- np.exp (-2*v0*t)) / ( 1+ np.exp (-2*v0*t)))
    return v  
velocity_data= [] 
altitude_data= [] 
time = []
t = 0 
tau = 2.56 / v0
# creating xticks and yticks. If we do not change the time simulation, we do not need to change anything 
xticks = []
xtickstext = []
yticks = []
ytickstext = [] 
#creating xticks 
for i in range (0,6): 
    xticks.append (i)
    xtickstext.append (str(i)) 
xticks.append (tau)
xtickstext.append(r'$\tau$')
xticks.append (tau)
xtickstext.append(r'$\tau$')
# creating yticks 
for i in range (0,9): 
    yticks.append (i)
    ytickstext.append (str(i)) 
yticks.append (0.99*v0)
ytickstext.append ('99% v0')


if alpha >0 and beta >0:
    print (f' Predicted terminal velocity: {(alpha/beta) ** (1/2)} (m/s) ')
    while t < simulation_time: 
        velocity_data.append (v(t,alpha=alpha,beta=beta))
        time.append(t)
        t+= 20/1000
    plt.plot (time,velocity_data)

    plt.scatter ([tau], [0.99 * v0], s = 30, color = 'red')
    plt.text (tau+0.2, 0.99* v0-0.6, f'({tau:.1f}, {0.99 *v0:.1f})', color='black', backgroundcolor = 'white')

    plt.xticks (xticks, xtickstext)
    plt.yticks (yticks, ytickstext)

    plt.xlabel ('time (s)')
    plt.ylabel ('velocity (m/s)')
    plt.title(f'Analytical solution of v(t) with v0 = {v0:.2f} (m/s)')
    plt.grid(True)
    plt.show ()

else: 
    print ('Cannot rise')

