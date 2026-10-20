# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:08:18 2026

@author: james
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
a = 3.0
b = 1.0
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Define the system of ODEs
# y[0] = x, y[1] = v
def system(t, y):
    x, v = y
    dxdt = v
    dvdt = (b * np.cos(2 * t) - a) * x
    return [dxdt, dvdt]
def system_solver(system, t_span, y0, t_eval):
    sol = solve_ivp(system, 
                    t_span, 
                    y0, 
                    method='RK45',
                    t_eval=t_eval)
        
    return sol
for y0 in [(0,1), (1,0), (-1,1), (0.5,0.33)]:
    t_out,x,v = system_solver(system,[0,10],y0,t_eval)
    plt.plot(x, v, linewidth=1.5, label = f'(x₀,v₀) = {y0}')

# Time span for simulation

plt.figure(figsize=(10, 7))
# Initial conditions: start at x=1, v=0
#pairs = ((1,0), (0,1), (1, 1), (1/2, 0.33))
# Solve the ODE


# Plot the trajectory in the phase plane


plt.scatter([0], [0], color='black', zorder=5, marker='x', s=100, label='Fixed Point (0,0)')

plt.xlabel('x (Position)')
plt.ylabel('v (Velocity)')
plt.axhline(0, color='green', linewidth=0.5, linestyle='--', label='Nullcline @ v = 0')
plt.axvline(0, color='blue', linewidth=0.5, linestyle='--',label='Nullcline @ x = 0')
plt.legend()


#plots the x and v as positions of time
plt.figure(figsize=(10,7))
for y0 in [(0,1), (1,0), (-1,1), (0.5,0.33)]:
    t_out,x,v = system_solver(system,[0,10],y0,t_eval)
    plt.plot(t_out,v,label='v(t)')
    plt.plot(t_out,x,label='x(t)')
plt.legend()
plt.show()