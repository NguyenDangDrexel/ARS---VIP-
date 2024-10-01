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
P0 = 101325 # Pa 


def constant_volume_balloon(h: float, initial_params: list) -> dict: 
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
    
    if len(initial_params) != 4:
        print("Initial parameters should be a list with 4 elements: [P0, n0, T0, delta_P].\n Please use help(constant_volume_balloon) to see the instructions")
        return None
        
    p0,n0,T0,delta_P = initial_params 

    # Pressure 
    P = Pressure(h) + delta_P 

    # Mole of Helium 
    n = n0

    # Determine volume of the balloon 
    V0 = (n0 * R * T0) / P0
    V = V0 

    # Determine the area of the balloon 
    A = np.pi * ((3/4) * (V/np.pi))**(2/3) 

    # Determine the radius of the balloon 
    r = ((3/4) * V / np.pi) ** (1/3)

    # Temperature 
    T = (P * V) / (n * R)
    
    state = {
        'Pressure': P, 
        'Volume': V,
        'Temperature': T, 
        'Area': A,
        'Radius': r,
        'Number of moles': n
    }  
    
    return state 

# Example usage

def iosthermal_balloon(h: float, initial_params: list) -> dict: 
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
    
    if len(initial_params) != 4:
        print("Initial parameters should be a list with 6 elements: [P0, n0, T0, A0, r0, delta_P].\n Please use help(constant_volume_balloon) to see the instructions")
        return None
        
    p0,n0,T0,delta_P = initial_params 

    # Temperature 
    T = T0 

    # Mole of Helium 
    n = n0

    # Pressure 
    P = Pressure (h) + delta_P 
    
    # Determine volume of the balloon 
    V0 = n0*R*T0/ P0
    V = V0 * (P0/P)

    # Determine the area of the balloon 
    A = np.pi * ((3/4) * (V/np.pi))**(2/3) 

    # Determine the radius of the balloon 
    r = ((3/4) * V / np.pi) ** (1/3)

    state = {
        'Pressure': P, 
        'Volume': V,
        'Temperature': T, 
        'Area': A,
        'Radius': r,
        'Number of moles': n
    }  
    
    return state 

# Example usage


def adiabatic_balloon(h: float, initial_params: list) -> dict: 
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
    
    if len(initial_params) != 4:
        print("Initial parameters should be a list with 6 elements: [P0, n0, T0, A0, r0, delta_P].\n Please use help(constant_volume_balloon) to see the instructions")
        return None
        
    p0,n0,T0,delta_P = initial_params  

    

    # Mole of Helium 
    n = n0

    # Pressure 
    P = Pressure (h) + delta_P 
    
    # Determine volume of the balloon 
    V0 = n0*R*T0/ P0
    V = (P0/Pressure(h)) ** (1/gamma)* V0

    # Temperature 

    T = P*V / (n*R)

    # Determine the area of the balloon 
    A = np.pi * ((3/4) * (V/np.pi))**(2/3) 

    # Determine the radius of the balloon 
    r = ((3/4) * V / np.pi) ** (1/3)

    state = {
        'Pressure': P, 
        'Volume': V,
        'Temperature': T, 
        'Area': A,
        'Radius': r,
        'Number of moles': n
    }  
    
    return state 
def simultaneous_temperature_balloon(h: float, initial_params: list) -> dict: 
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
    
    if len(initial_params) != 4:
        print("Initial parameters should be a list with 6 elements: [P0, n0, T0, A0, r0, delta_P].\n Please use help(constant_volume_balloon) to see the instructions")
        return None
        
    p0,n0,T0,delta_P = initial_params 

   

    # Mole of Helium 
    n = n0

    # Pressure 
    P = Pressure (h) + delta_P 
    # Determine volume of the balloon 
    V0 = n0*R*T0/ P0
    V = (P0/Pressure(h)) ** (1/gamma)* V0

    # Temperature 
    T = P*V / (n*R)
    
    

    

    # Determine the area of the balloon 
    A = np.pi * ((3/4) * (V/np.pi))**(2/3) 

    # Determine the radius of the balloon 
    r = ((3/4) * V / np.pi) ** (1/3)

    state = {
        'Pressure': P, 
        'Volume': V,
        'Temperature': T, 
        'Area': A,
        'Radius': r,
        'Number of moles': n
    }  
    
    return state 
