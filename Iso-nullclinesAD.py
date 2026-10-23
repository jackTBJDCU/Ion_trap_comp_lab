# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:56:26 2026

@author: Aaron Dignam
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def coupled(x,y):
    dx = x**2 + x*y - 2*x
    dy = x*(y**2) + 3*y
    
    return dx, dy

def nullclines(x,y):
    nx = 2 - y
    
    if x == 0:
        ny = np.nan
    else:
        ny = -3/x
    
    return nx, ny

def rhs(t, state):
    # solve_ivp calls fun(t, state), so this wraps coupled(x, y)
    x, y = state
    return coupled(x, y)

def main():
    coords = np.linspace(-4, 4, 21)
    x, y = np.meshgrid(coords, coords)    
    dx,dy = coupled(x, y)
    
    plt.quiver(x,y,dx,dy, color='gray')
    plt.streamplot(x, y, dx, dy, color='lightgray')
    
    # Plotting zero nullclines
    ts = np.linspace(-4, 4, 401)          # 401 points so t = 0 is included
    nx_vals = []
    ny_vals = []
    for t in ts:
        nx, ny = nullclines(t, t)
        nx_vals.append(nx)
        ny_vals.append(ny)

    
    #plotting axis
    plt.axvline(0, color='k') # plots vertical axis
    plt.axhline(0, color='k') # plots horizontal axis
    
    plt.plot(nx_vals,ts, '--', label='dx/dt = 2-y')
    plt.plot(ts, ny_vals, label='dy/dt = -3/x')
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.title('Nullclines for f(x,y) and g(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    #fixed points
    # Fixed values found by hand
    fp_vals = np.array([[0,2,-1,3], [0,0,3,-1]])

    plt.plot(fp_vals[0],fp_vals[1], 'ko', label='fixed points')
    
    plt.legend(loc='upper right', fontsize=7)   
    plt.show()
    plt.close()
    
    t_eval = np.linspace(0, 10,400)  # Points at which output will be evaluated
    
    # Exact Solution
    
    for fx, fy in zip(fp_vals[0],fp_vals[1]):
        y0 = [fx + 0.1, fy + 0.1]
        # Calls the method integrate.solve_ivp()
        result = integrate.solve_ivp(fun=rhs,  # The function defining the derivative
                                     t_span=(0, 10),  # Initial and final times
                                     y0=y0,  # Initial state
                                     method="RK45",  # Integration method
                                     t_eval=t_eval,
                                     args=())  # Time points for result to be defined at
        
        # Read the solution and time from the result array returned by Scipy
        xs, ys = result.y
        t = result.t
        
        plt.figure()
        plt.plot(t, xs, label='x(t)')
        plt.plot(t, ys, label='y(t)')
        plt.title(f'Start ({y0[0]:.1f}, {y0[1]:.1f}) near fixed point ({fx}, {fy})')
        plt.xlabel('t')
        plt.ylabel('x, y')
        plt.legend()
        plt.show()
    
    


if __name__ == '__main__':
    main()