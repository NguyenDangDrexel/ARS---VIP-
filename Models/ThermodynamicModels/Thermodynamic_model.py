import numpy as np 
import sys

# In order to import the file in the parent directory, we add the directory containing the parent file to the sys.path 
# setting path
path = r"C:\New folder\Drexel\2023\Courses\Summer 2024 - SGN\VIP program - balloon project\Weather-Balloon-Drexel\NEBP_project"
sys.path.append(path)
 
from Models.Atmospheric_models.AtmosphericModel import Pressure, Temperature, Density 


g0 = 9.80665 #(m/s^2)
R = 8.31413# ( J/mol*K) universal constant 
M = 0.0286944  # (g/mol) - molar mass of dried gas 
mu_0 = 1.77 * 10 ** (-5) # Dynamic density (Pa-s)
#constant of adiabatic process 
gamma = 1.4 
# from  0 <= h <= 11000:  T = a1 +b1h 
a1 = 288.04 # Kelvin 
b1 = -0.00649 # (1/m)
# from 11000 < h < 25000: T = 216.54 (K)
a2 = 216.54 
b2 = 0 
# from 25000 < h < 35000: T = a3 + b3h 
a3 = 141.79
b3 = 0.00299
P0 = 101325 # preesure at sea level 

# The range of altitude 
H0 = 0 
H1 = 11000 
H2 = 25000
H3 = 35000   



# Input: the initial value of volume 
def Volume_constant (V0): 
    V = V0 
    return V 
#Input: the volume of the balloon 
def Area_constant (V0):
    A =np.pi * ((3/4) * (Volume_constant(V0)/np.pi))**(2/3)
    return A 
def Radius_constant (V0):
    R = (3/4 * Volume_constant (V0) / np.pi) ** (1/3)
    return R 
# Radius function calculated from the Volume equation



#Isothermal model (T_in = constant), Input: heigh, initial volume
def Volume_isothermal (h,P0= P0, V0=1,T0=a1):
    V = V0 * ((P0)/(Pressure(h)))
    return V 
def Radius_isothermal (h,V0 = 1): 
    R = Volume_isothermal(h,V0=1) ** (1/3)
    return R 
def Area_isothermal (h,V0):

# Adiabatic model 
    A =np.pi * ((3/4) * (Volume_isothermal(h,V0 =V0)/np.pi))**(2/3)
    return A 
def Volume_adiabatic (h,P0=P0,V0=1):
    V = (P0/Pressure(h)) ** (1/gamma)* V0
    return V 
def Area_adiabatic (h,V0):
    A =np.pi * ((3/4) * (Volume_adiabatic(h,V0 =V0)/np.pi))**(2/3)
    return A 
def Radius_adiabatic (h,V0 = 1): 
    R = Volume_adiabatic(h,V0=1) ** (1/3)
    return R 

# this equation is taken from Henrique Yago paper 
def Volume_data (h, rf,constant, ri =Radius_constant(V0=1)):
    V = 4/3 * np.pi * (constant + (ri - rf)/(Pressure(0) - Pressure(25000)) * Pressure(h)) ** (3)
    return V 

def Volume_out (h,P0=P0,V0=1,T0=a1):
    V = V0 * ( (P0)/(Pressure(h)) ) * ( (Temperature(h))/(T0) )
    return V 
def Area_out (h,V0):
    A =np.pi * ((3/4) * (Volume_out(h,V0 =V0)/np.pi))**(2/3)
    return A 
def Radius_out (h,V0 = 1): 
    R = Volume_out (h,V0=1) ** (1/3)
    return R 
def Temperature_adiabatic (h,V0=1, T0= a1): 
    T= T0 * (Pressure(0) / Pressure(h)) ** ((1-gamma)/gamma )
    return T 

