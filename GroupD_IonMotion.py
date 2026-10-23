import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def function(v,x,b,t,a):
    dx = v
    dy = x*b*np.cos(2*t)-x*a
    
    return dx,dy

def solve(t, state, a, b):
    v, x = state
    return function(v, x, b, t, a)

def main():
    #define grid
    coords = np.linspace(-4, 4, 21)
    x, y = np.meshgrid(coords, coords)    
    
    
    #initial values
    v = y
    b = 1
    a = 1
    t = 2
    dx,dy = function(v,x,b,t,a)

    
    plt.quiver(x,y,dx,dy, color='gray')
    plt.streamplot(x, y, dx, dy, color='lightgray')
    #plotting nullclines
    plt.axvline(0, color='k', label='x and y nullcline') # plots vertical nullcline
    plt.axhline(0, color='k') # plots horizontal nullcline
    plt.plot(0,0,'bo', label='fixed point')
    plt.legend(loc='upper right', fontsize='7')
    plt.title('Motion of the Ion within the trap')
    plt.show()
    plt.close()
    '''
    SOLVER FOR THE FUNCTION
    '''
    t_eval = np.linspace(0,10,400)
    y0 = [1,0]
    #Solving to plot and determine behaviours
    # Calls the method integrate.solve_ivp()
    result = integrate.solve_ivp(fun=solve,  # The function defining the derivative
                                 t_span=(0, 10),  # Initial and final times
                                 y0=y0,  # Initial state
                                 method="RK45",  # Integration method
                                 t_eval=t_eval,
                                 args=(a,b))  # Time points for result to be defined at
    
    # Read the solution and time from the result array returned by Scipy
    xs, ys = result.y
    t = result.t
    
    plt.figure()
    plt.plot(t, xs, label='x(t)')
    plt.plot(t, ys, label='y(t)')
    plt.title(f'Ion position and speed as a function of time')
    plt.xlabel('Time')
    plt.ylabel('x(t), y(t)')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    main()