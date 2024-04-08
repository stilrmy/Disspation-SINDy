import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from numpy import sin,cos

# Define the equations of motion
def equations_of_motion(t, y, g, l1, l2, m1, m2, b1, b2, tau0, omega1, omega2, phi):
    theta1, theta2, theta1_dot, theta2_dot = y
    # The actual equations of motion based on your provided expressions would go here
    # For demonstration, we'll use placeholders for theta1_dot_dot and theta2_dot_dot
    theta1_dot_dot = -b1*l2*theta1_dot/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) + b2*l1*theta2_dot*cos(theta1 - theta2)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) - g*l1*l2*m1*sin(theta1)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) - g*l1*l2*m2*sin(theta1)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) + g*l1*l2*m2*sin(theta2)*cos(theta1 - theta2)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) - l1**2*l2*m2*theta1_dot**2*sin(theta1 - theta2)*cos(theta1 - theta2)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) - l1*l2**2*m2*theta2_dot**2*sin(theta1 - theta2)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) - l1*tau0*sin(omega2*t + phi)*cos(theta1 - theta2)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2) + l2*tau0*cos(omega1*t + phi)/(l1**2*l2*m1 - l1**2*l2*m2*cos(theta1 - theta2)**2 + l1**2*l2*m2)
    theta2_dot_dot = b1*l2*m2*theta1_dot*cos(theta1 - theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) - b2*l1*m1*theta2_dot/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) - b2*l1*m2*theta2_dot/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) + g*l1*l2*m1*m2*sin(theta1)*cos(theta1 - theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) - g*l1*l2*m1*m2*sin(theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) + g*l1*l2*m2**2*sin(theta1)*cos(theta1 - theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) - g*l1*l2*m2**2*sin(theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) + l1**2*l2*m1*m2*theta1_dot**2*sin(theta1 - theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) + l1**2*l2*m2**2*theta1_dot**2*sin(theta1 - theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) + l1*l2**2*m2**2*theta2_dot**2*sin(theta1 - theta2)*cos(theta1 - theta2)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) + l1*m1*tau0*sin(omega2*t + phi)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) + l1*m2*tau0*sin(omega2*t + phi)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2) - l2*m2*tau0*cos(theta1 - theta2)*cos(omega1*t + phi)/(l1*l2**2*m1*m2 - l1*l2**2*m2**2*cos(theta1 - theta2)**2 + l1*l2**2*m2**2)
    return [theta1_dot, theta2_dot, theta1_dot_dot, theta2_dot_dot]

# Initial conditions
theta1_0 = 0  # Initial angle for pendulum 1
theta2_0 = 0  # Initial angle for pendulum 2
theta1_dot_0 = 0  # Initial angular velocity for pendulum 1
theta2_dot_0 = 0  # Initial angular velocity for pendulum 2
y0 = [theta1_0, theta2_0, theta1_dot_0, theta2_dot_0]

# Parameters
g = 9.81  # Acceleration due to gravity
l1 = l2 = 1  # Lengths of the pendulum arms
m1 = m2 = 1  # Masses of the pendulums
b1 = b2 = 0.1  # Damping coefficients
tau0 = 2  # Amplitude of the external torque
omega1 = 0.5 * np.pi  # Frequency of the external torque
omega2 = 0.2 * np.pi
phi = 0  # Phase of the external torque

# Time span
t_span = (0, 10)
high_res_frames = 900
t_eval = np.linspace(*t_span, high_res_frames)

# Solve the system of differential equations
sol = solve_ivp(equations_of_motion, t_span, y0, args=(g, l1, l2, m1, m2, b1, b2, tau0, omega1, omega2, phi), t_eval=t_eval)

# Plotting and animation setup
fig, ax = plt.subplots()
ax.set_xlim((-2.2, 2.2))
ax.set_ylim((-2.2, 2.2))

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    frame_index = int(i * (high_res_frames / 300))
    x1 = l1 * np.sin(sol.y[0, frame_index])
    y1 = -l1 * np.cos(sol.y[0, frame_index])
    x2 = x1 + l2 * np.sin(sol.y[1, frame_index])
    y2 = y1 - l2 * np.cos(sol.y[1, frame_index])
    line.set_data([0, x1, x2], [0, y1, y2])
    return line,

total_real_time_frames = 30 * t_span[1]

ani = FuncAnimation(fig, animate, frames=total_real_time_frames, init_func=init, blit=True)

ani.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

print("Animation saved as 'double_pendulum.mp4'.")
