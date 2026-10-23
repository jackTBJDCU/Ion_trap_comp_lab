import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def function(v,x,b,t,a):
    dx = v
    dy = x*b*np.cos(2*t)-x*a
    
    return dx,dy

def solve(t, state, a, b):
    x, v = state                      # state order is [x, v]
    dx, dv = function(v, x, b, t, a)
    return [dx, dv]

def main():
    # (a, b) pairs to compare
    params = [(3, 0.5), (2, 1), (1, 1), (1, 2)]
    
    #define grid
    coords = np.linspace(-4, 4, 21)
    x, y = np.meshgrid(coords, coords)    
    
    t_eval = np.linspace(0,10,400)
    y0 = [1,0]                        # initial state [x(0), v(0)]
    t_snap = 2                        # time at which the vector field is drawn
    
    for a, b in params:
        plt.figure(figsize=(12,5))
        plt.suptitle(f'Motion of the Ion within the trap: a = {a}, b = {b}')
        
        # Left panel: vector field snapshot at t = t_snap
        plt.subplot(1,2,1)
        v = y
        dx,dy = function(v,x,b,t_snap,a)
        plt.quiver(x,y,dx,dy, color='gray')
        plt.streamplot(x, y, dx, dy, color='lightgray')
        #plotting nullclines
        plt.axvline(0, color='k', label='x and y nullcline') # plots vertical nullcline
        plt.axhline(0, color='k') # plots horizontal nullcline
        plt.plot(0,0,'bo', label='fixed point')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Vector field at t = {t_snap}')
        plt.legend(loc='upper right', fontsize='7')
        
        # Right panel: solve for this a and b
        result = integrate.solve_ivp(fun=solve,
                                     t_span=(0, 10),
                                     y0=y0,
                                     method="RK45",
                                     t_eval=t_eval,
                                     args=(a,b))
        
        xs, ys = result.y
        t = result.t
        
        plt.subplot(1,2,2)
        plt.plot(t, xs, label='x(t)')
        plt.plot(t, ys, label='y(t)')
        plt.title('Ion position and speed as a function of time')
        plt.xlabel('Time')
        plt.ylabel('x(t), y(t)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    
if __name__ == '__main__':
    main()