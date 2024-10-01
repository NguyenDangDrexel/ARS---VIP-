import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os  
import random 
from tqdm.auto import tqdm 
from scipy.integrate import odeint 
from scipy.signal import medfilt 

# ---------------------------- READ ME --------------------------------------------
"""
Status: on-going 
Tasks: 
1. Altitude sensor random noise 
2. Vertical velocity random noise 
3. Resolution of altitude sensor 
4. Create the function for control rules 
"""
#-------------------------------- import neccessary modules ------------------ 

import sys
path = r"C:\Drexel\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project"
sys.path.append(path) 
from Models.ThermodynamicModels import Balloon
from Models.Atmospheric_models.AtmosphericModel import Pressure, Temperature, Density 
#-------------------------------------------------------------------------------------------- 

# --------------------------------physics parameters -----------------------------

# Molar mass of Helium (kg/mol) <=> rho (kg/m^3)
muy_he = 0.004 #(kg/mol)
# Density of Helium 
rho_he = 0.1784 # kg/m^3 
# Drag coefficient  
Cd = 0.581
# Gas constant 
R = 8.31 
# Gravitational constant 
g = 9.8 

# -------------------------------- balloon parameters -----------------------------
# Surface tension pressure; from 100 to 200 Pa 
difference_pressure = 200 # (Pa)
# Initial volume of Helium - 
V0 = 15.05 # m3
# mass of the balloon (helium inside + payload + latex)
m = 7
# Initial mol of helium 
initial_mol_helium = Pressure (0)* V0 / (R *Temperature (0))

# -------------------------------- air_drag parameters -----------------------------
C = 0.5 # m_air = C * rho * V  
include_m_air  = True # True if we want to 

# -------------------------------- venting parameters -----------------------------

# Diameter of venting system (m): 
diameter = 0.095 
# Area of the venting 
area_of_orifice = 1/4 * np.pi * diameter**2 * 1 

# Discharged rate constant - needed 
discharged_constant = 0.1  # 
# -------------------------------- simulation parameters -----------------------------
# case 1: constant volume balloon 
# case 2: constant temperature 
# case 3: adiabatic 
# case 4: temperature inside = temperature ouside 

initial_params = [101325,626,293, 0] # [P0,n0,T0,delta_P]

#Start at the ground level, turn on the controller when it hits some altitudel; V0 (neck lift, mass,...) 

# case 4: T_in = T_out 
# Case 3: think about the adiabatic process 
# For now, do 2,4 
def venting(y:list, t,u:float) -> list :
    """
    parameters 

    Returns: list 
        return the velocity, acceleration, dn/dt at time t 
        [ dydt = [xdot, xddot, ndot] ]
    
    Problem: since the 
    
    """
    # vector state : y = [x,xdot,n]
    x = y[0] 
    xdot = y[1]
    n = y[2] # mol of Helium 
    
    # Choose case to do simulation 
    case = 2 
    if case == 2: 
        T = Balloon.iosthermal_balloon (x,initial_params= initial_params)['Temperature']
        P = Balloon.constant_volume_balloon (x,initial_params= initial_params)['Pressure']
        V = R * T *n /P # ONLY for this case 
        A = Balloon.iosthermal_balloon (x,initial_params= initial_params)['Area']
    elif case == 3: # this case is not accurate => do not use 
        T = Balloon.adiabatic_balloon (x,initial_params= initial_params)['Temperature']
        P = Balloon.constant_volume_balloon (x,initial_params= initial_params)['Pressure']
    elif case  == 4: 
        T = Balloon.simultaneous_temperature_balloon (x,initial_params= initial_params)['Temperature']
        P = Balloon.constant_volume_balloon (x,initial_params= initial_params)['Pressure']


    # Use ideal gas law to define volume 
    # V = n*R*T/ P  # NOTE: in this line, n changes => cannot use the V from Balloon, which is the closed balloon 

    # air_drag 
    m_air = C * Density(x) * V 
    
    # F = ma - dynamic equation 
    rel_velocity = xdot + 0 
    if xdot > 0: 
        xddot = (-m * g + Density(x) * V * g) / (m + m_air) - 0.5 * Density(x) * rel_velocity**2 * A * (Cd+random_vertical_velocity (amplitude= 0.2))  / (m + m_air)  
    else: 
        xddot = (-m * g + Density(x) * V * g) / (m + m_air) + 0.5 * Density(x) * xdot**2 * A * Cd / (m + m_air) 
    # adding noise to Cd to simulate vertical movement of air - 1 option (xdot-noise) **2. This noise can be random, or the function of altitude/time 
    # Define the function to calculate volume discharged rate 
    Q = u*-discharged_constant * area_of_orifice * ((2 * R) / (muy_he) * difference_pressure) ** (1/2) * ((Temperature(x)) / (Pressure(x))) ** (1/2)
    # Define ndot = Q * rho / muy 
    ndot = (Q * rho_he)/ muy_he
    # P, T, n => V => diameter (drag equation?) ??? 
    # dydt = [x,xdot,n]
    dydt = [xdot , xddot, ndot] 
    return dydt 

# -------------------------------------------------------------------------------------------------------
# Resolution of sensor 
def solution_sensor (h, resolution = 0.2): 
    """
    This function simulates the resolution of the altitude sensor. 
    The default resolution is 0.2 meter 
    Params: 
        h: noised altitude 
        resolution: resolution of the sensor 

    Returns: 
        h_resolution 
    """
    h_resolution = round (h/resolution) * resolution
    return h_resolution
# ------------------------------------------------------------------------------------------------------
# Random noise of sensor 
def random_noise_sensor (h:float,amplitude:int):
    """
    This function simulates the random noise of sensor by using random.randint (begin,end) method. 
    This method generates random number in the range begin -> end.
    Params: 
        h: altitude from odeint (), which is the "true" altitude of the balloon 
        amplitude
    Return: 
        h_noise: altitude + random noise 
    """
    h += random.randint (-1,1) * amplitude 
    return h 
# ---------------------------------------------------------------------------------------------------------
# random vertical velocity 
def random_vertical_velocity (amplitude = 1):
    """
    This function generates the vertical velocity
    Params: 
        amplitude: default value = 1 
    """

    vertical_velocity = random.randint (0,1) * amplitude
    return vertical_velocity

# ------------------------------------------------------------------------------------------------------ 
# PID 
def PID (ODEINT_initial_params: list, time_params:list, PID_params:list, control_params:list, altitude_setpoints:list  ) -> list: 
    """ 
    ODEINT_initial_params:
        list: [x0,v0,n0]
    
    time_params
        list: [time_simulation,time_step]
    
    PID_params: 
        list: [kp,ki,kd]
    control_params: 
        list:[control_interval, target_setpoint, min_time_of_valve]

    altitude_setpoints = [al_min, al_target,al_max ]
        
    Returns: 
    """
    y0 = ODEINT_initial_params 

    # PID time parameters

    simulation_time = time_params[0]                                  # final time for simulation;  
    delta_t = time_params[1]                                              # time step for odeint - do not change 
    nsteps = int (simulation_time/delta_t) + 1             # number of time steps 
    ts = np.linspace(0, simulation_time, nsteps)           # linear vector space
    # control interval - seconds 
    control_interval = control_params[0] # need to tune PID again if we change this parameter 


    # lists for storing the results
    venting_status = np.zeros(nsteps) # u = valve % open
    v_list = np.zeros(nsteps)
    sps = np.zeros(nsteps)
    altitude_lx = np.zeros(nsteps)
    venting_time = np.zeros (nsteps) 

    # PID control parameters 
    Kp = PID_params[0]
    Ki = PID_params [1] # it seems that we should not tune this parameters. When the value is too small => no effect. When it's big => so much 
    Kd =  PID_params [2]
    ubias = 0 



    altitude_sp_min = altitude_setpoints[0]
    altitude_sp = altitude_setpoints[1]
    altitude_sp_max = altitude_setpoints[2] 
# time from min to setpoint altitude 
    avg_velocity = 7 
    tau = int ((altitude_sp - altitude_sp_min)/ avg_velocity)

    error_prev = 0
    u = ubias 
    sum_int = 0 
    v0 = y0[1]

    min_time_of_valve = control_params[2] # when the u* control_interval <3 => we don't open the valve => avoid opeening the valve too fast 

    # u: 0-1. 0.065. 
    # high altitude => ice => moving the valve ??? 

    for i in tqdm (range(nsteps - 1)):# 1 seconds. 0,1,2,3,4 ..., 30,31,32,...,60  
        y = odeint(venting, y0, [0, delta_t], args=(u,) )

        # online activate the PID at 0,20,40,60, ...  
        if y[-1][0] < altitude_sp_min: 
            # rise up naturally, do not apply PID 
            u = 0 
        elif altitude_sp_min <= y[-1][0] < altitude_sp:
            velocity_sp = 1 *(avg_velocity- abs ((y[-1][0] - altitude_sp_min))/tau )  + control_params[1]  # this setpoint change with time 
            # trigger PID -  we want to control the balloon - setpoint decrease (as the function of altitude)from avg-velocity to 0 
            if y[-1][1] >=2: 
                control_interval = control_interval
            else: 
                control_interval = 3 
            if i % control_interval == 0:
                error = v0-velocity_sp
                sum_int += error * control_interval
                d_error = (error - error_prev) / control_interval 
                u = ubias + Kp * error + Ki * sum_int + Kd * d_error
                # setting the threshold, preventing the extremely small control signal 
                if u * control_interval >min_time_of_valve: 
                    
                    u = max(0, min(1, u))  # constrain u to 0-100%
                elif u* control_interval <min_time_of_valve: 
                    u = 0
                error_prev = error
        
                
                # store the venting_time: reflect the real status of the valve 
                time = round (u * control_interval) 
                if time >0:

                    print (f'open the valve for {time} seconds, at time {i}')

                for t in range (time): 
                    venting_time[t+i] = 1 
            
        else: 
            velocity_sp = 0 
            # trigger PID with velecity_sp =0 
            if y[-1][1] >=2: 
                control_interval = control_interval
            else: 
                control_interval = 3 
            if i % control_interval == 0:
                error = v-velocity_sp
                sum_int += error * control_interval
                d_error = (error - error_prev) / control_interval
                
                u = ubias + Kp * error + Ki * sum_int + Kd * d_error
                u = max(0, min(1, u))  # constrain u to 0-100%
                error_prev = error
                time = round (u * control_interval) 
                if time > 0: 
                    print (f'open the valve for {time} seconds, at time {i}')
                for t in range (time): 
                    venting_time[t+i] = 1 
        
            
        # When i = t; odeint()  gives us the state of the balloon at time t+1 
        y = odeint(venting, y0, [0, delta_t], args=(u,) )
        #---------------- without noise --------------- 
        # y0 = y[-1]  # update initial conditions. y = [x,xdot,n] 

        # ----------- add Gaussian noise to  ---------------------------------------------------------------  
            # Extract the altitude at t+1 
        altitude = y[-1][0]                                   # x before adding noise 
        altitude = random_noise_sensor (altitude,amplitude = 0.2)

                        # the range of noise from -1 to 1 
        # Calculate the average velocity from time (t -> t+1) from noised altitude 
        v0 = (altitude-y0[0])/delta_t                        # y0 is the state vector at time t  
        # altitude at time t-1. 
        # ------------------ use Median filtering to filter the V0 

        # update the state vector at time t + 1. We need to feed this vector into odeint () to get vector at time t+2 
        y0 =[altitude,v0,y[-1][2]]
        v = y[-1][1]                                        # get the velocity at time t+1 from odeint ()
# ------------------------------- store the values for plotting ---------------------------------------------- 

        # Store the value to the lists for plotting 
        altitude_lx [i+1] = altitude 
        v_list[i + 1] = v0 # store the velocity for plotting
        if y[-1][0] < altitude_sp_min:
            sps[i + 1] = 0
        else: 
            sps[i + 1] = velocity_sp # store the setpoint for plotting
        venting_status[i + 1] = u 
    return altitude_lx, v_list,sps,venting_status,ts
# --------------------------------------------------------------- 

ODEINT_initial_params = [0,0,initial_mol_helium]# [x,xdot,n]
time_params = [4000,1] #[time_simulation,time_step] 
PID_params = [0.4,0,0.1] # [kp,ki,kd]
control_params = [20,2,3] # [control_interval, velocity_final_setpoint, min_time_of_valve] 
altitude_setpoints = [23000,28000,30000] # [al_min, al_target,al_max ] 


altitude_lx, v_list,sps,venting_status,ts = PID (ODEINT_initial_params,time_params,PID_params,control_params,altitude_setpoints)

# plotting 


# Create subplots
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(10, 8))
start_time = 0
end_time = 5000
simulation_time = time_params[0]
if end_time <= simulation_time: 
    end_time = end_time 
else: 
    end_time = simulation_time


# Plot the first set of data on the first subplot
ax1.plot(ts[start_time:end_time], v_list[start_time:end_time], label='velocity')
ax1.plot (ts[start_time:end_time],sps[start_time:end_time],label = 'setpoint')
ax1.set_xlabel('Time')
ax1.set_ylabel('Velocity')
ax1.legend()
ax1.grid(True)

# Plot the second set of data on the second subplot
ax2.plot(ts[start_time:end_time], venting_status[start_time:end_time], label='Venting time during the sampling time', color='orange')
ax2.set_xlabel('Time')
ax2.set_ylabel('venting -PID ')
ax2.legend()
ax2.grid(True)
ax3.plot(ts[start_time:], altitude_lx[start_time:], label='altitude', color='red')
ax3.set_xlabel('Time')
ax3.set_ylabel('altitude')
ax3.legend()
ax3.grid(True)

# ax4.plot(ts[start_time:], venting_time[start_time:], color='red')
ax4.set_xlabel('Time')
ax4.set_ylabel('ventin - real life')
ax4.legend()
ax4.grid(True)
# Adjust the layout to prevent overlap
plt.tight_layout()

#control-interval - 20 seconds 
# control signal = 0.25 => 20 * 0.25 = 5 seconds 
# There is not waiting time 
# 0 -20: 1 => 20 s
# 21-40: 0.5 => 10 seconds 
# when 1 (m/s) => change control strategy: maybe 
# tuning PID 
# Show the plots
plt.show() 



