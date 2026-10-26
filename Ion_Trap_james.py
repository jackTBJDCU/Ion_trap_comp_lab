import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
a = 3.0
b = 1.0
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

def system(t, y):
    x, v = y
    dxdt = v
    dvdt = (b * np.cos(2 * t) - a) * x
    return [dxdt, dvdt]

def system_solver(system, t_span, y0, t_eval):
    sol = solve_ivp(system, t_span, y0, method='RK45', t_eval=t_eval)
    return sol.t, sol.y[0], sol.y[1]

initial_conditions = [(0,1), (-1,1), (0.5,0.33)]
colors = plt.cm.viridis(np.linspace(0, 1, len(initial_conditions)))

# ---- Graph 1: x(t) and v(t) vs time, same colour per initial condition ----
plt.figure(figsize=(10, 7))
for y0, col in zip(initial_conditions, colors):
    t_out, x, v = system_solver(system, t_span, y0, t_eval)
    plt.plot(t_out, x, color=col, linestyle='-',  linewidth=1.5,
             label=f'x(t),  (x₀,v₀)={y0}')
    plt.plot(t_out, v, color=col, linestyle='--', linewidth=1.5,
             label=f'v(t),  (x₀,v₀)={y0}')

plt.xlabel('T (Time)')
plt.ylabel('x(t), v(t)')
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()

# ---- Graph 2: phase plane ----
plt.figure(figsize=(10, 7))
for y0, col in zip(initial_conditions, colors):
    t_out, x, v = system_solver(system, t_span, y0, t_eval)
    plt.plot(x, v, color=col, linewidth=1.5, label=f'(x₀,v₀) = {y0}')

plt.scatter([0], [0], color='black', zorder=5, marker='x', s=100,
            label='Fixed Point (0,0)')
plt.xlabel('x (Position)')
plt.ylabel('v (Velocity)')
plt.axhline(0, color='green', linewidth=0.5, linestyle='--', label='Nullcline @ v = 0')
plt.axvline(0, color='blue',  linewidth=0.5, linestyle='--', label='Nullcline @ x = 0')
plt.legend()
plt.tight_layout()
plt.show()