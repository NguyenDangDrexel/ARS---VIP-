import sys
from scipy.integrate import odeint 
import numpy as np 
from scipy.optimize import curve_fit 
"""
Status: now this one can run successfully 
Tasks: clean the code: create function + change something 


"""
import sys
path = r"C:\Drexel\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project"
sys.path.append(path) 
from Models.ThermodynamicModels import Balloon
from Models.Atmospheric_models.AtmosphericModel import Pressure, Temperature, Density 
 # [P0,n0,T0,delta_P]

#Drag coefficient  
Cd = 0.44 
C = 0.5 # m_air = C * rho * V 
g = 9.8 
def numerical_solultion_altitude_data (t,m,n0 ): 
    case = 4
    initial_params = [101325,n0,293, 0]
    # IMPORTANT: the form of the function matter. we need to put the highest order on left side, and the rest on the right side 
    def function (y,t):
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


        if xdot >0: 
            xddot = (-m*g + Density(x)*V *g)/ (m +m_air)  - 1/2 *Density(x) * xdot**2 * A*Cd /(m + m_air)  
        else: 
            xddot = (-m*g + Density(x)*V *g)/ (m +m_air)  + 1/2 *Density(x) * xdot**2 * A*Cd /(m + m_air) 
        if x <=35000:

            return xdot,xddot 
        else:
            return 0,0

    solution2 = odeint (function, y0= [42,0], t =t) 

    numerical_altitude_data = solution2[:,0]
    # numerical_ascend_rate_data = solution2[:,1]

    return numerical_altitude_data


path = r"C:\Drexel\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project\Data\20240408.xlsx"
import pandas as pd 
df = pd.read_excel (path)

# Convert time to seconds (same as before)
initial = pd.to_timedelta(df[' time [h:m:s GMT]'].iloc[0])
time_converted_lx = [(pd.to_timedelta(df[' time [h:m:s GMT]'].iloc[i]) - initial).total_seconds() for i in range(len(df[' time [h:m:s GMT]']))]
df[' time [h:m:s GMT]'] = time_converted_lx

xdata_train = df [' time [h:m:s GMT]'][1250:2000].tolist ()
xdata_new_train = []
ydata_train = df[' altitude (from iMet PTU) [km]'][1250:2000]*1000

for i in range (len (xdata_train)):
    xdata_new_train.append (xdata_train[i]-xdata_train[0])

bounds = ([0, 0], [7, 700]) # ([lower bounds],[upper bounds])
# popt,cov = curve_fit (numerical_solultion_altitude_data,xdata_new,ydata, bounds = bounds)
popt, pcov = curve_fit (lambda t,m,n0 :numerical_solultion_altitude_data (t,m,n0),xdata_new_train,ydata_train,bounds = bounds)
m_optimzed, n_optimized = popt[0],popt[1]

print (f'mass:{m_optimzed}, number of mole: {n_optimized}')

# ----------------------- 
xdata_full = df [' time [h:m:s GMT]'][1250:2500]
y_data_full = df[' altitude (from iMet PTU) [km]'][1250:2500]*1000
import matplotlib.pyplot as plt 
plt.plot (xdata_full,y_data_full)
plt.plot (xdata_full,numerical_solultion_altitude_data(xdata_full,m_optimzed,n_optimized), label = 'optimized')
plt.legend ()
plt.show () 




