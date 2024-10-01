import numpy as np 
# Define parameters
N = 100
a = 2  # Slope of the line
t = np.linspace(0, 10, N)  # 100 data points for x from 0 to 10

# Generate noise (normal distribution)
noise = np.random.normal(0, 1, 100)  # Mean 0, standard deviation 1

# Compute y = ax + noise
altitude = a * t + noise

S_xy = 0 
S_xx = 0  
S_x = 0 
S_y = 0 

# Manual LR: 
for index, altitude in enumerate (altitude): 
    x = t[index]
    y = altitude 
    S_xy += x*y
    S_x += x 
    S_y +=y 
    S_xx += x**2 

Sum_xx = S_xx - (S_x**2 / N) 
Sum_xy = S_xy - (S_x * S_y)/N 

velocity = S_xy / S_xx
print (velocity)

