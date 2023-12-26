import numpy as np
import random
import os
import csv
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoClip
# Pendulum rod lengths (m), bob masses (kg).

# The gravitational acceleration (m.s-2).
g = 9.81
tau = 0
attenuation_rate = 100

density = 200 #density of the bar in the image
def get_pendulum_data(n_ics,params):
    # The gravitational acceleration (m.s-2).
    g = 9.81
    t,X,Xdot = generate_pendulum_data(n_ics,params)

    data = {}
    data['t'] = t
    data['z'] = X[:,:2]
    data['dz'] = X[:,2:]
    data['ddz'] = Xdot[:,2:]

    #adding noise

    if params['adding_noise'] == True :
        if params['noise_type'] == 'image_noise':
            print('Adding noise to the pendulum data')
            print('noise_type: image noise')
            mu,sigma = 0,params['noiselevel']
            noise = np.random.normal(mu,sigma,data['x'].shape[1])
            for i in range(data['x'].shape[0]):
                data['x'][i] = data['x'][i]+noise
                data['dx'][i] = data['dx'][i] + noise
                data['ddx'][i] = data['ddx'][i] + noise

    return data






def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def cartPendulum_wrapper(params):
    def cartPendulum(t, y):
        M,m,R,k,d,b = params['M'],params['m'],params['R'],params['k'],params['d'],params['b']
        x,theta,x_t,theta_t = y
        x_tt = (m*R*theta_t**2*np.sin(theta)+m*g*np.sin(theta)*np.cos(theta)-k*x-d*x_t+b*theta_t*np.cos(theta)/R)/(M+m*np.sin(theta)**2)
        theta_tt = (-m*R*theta_t**2*np.sin(theta)*np.cos(theta)-(M+m)*g*np.sin(theta)+k*x*np.cos(theta)+d*x_t*np.cos(theta)-(1+M/m)*b*theta_t/R)/(R*(M+m*np.sin(theta)**2))

        return x_t,theta_t,x_tt,theta_tt
    return cartPendulum


def generate_pendulum_data(n_ics,params):
    if params['specific_random_seed'] == True:
        np.random.seed(params['random_seed'])
        random_seed = np.random.randint(low=1,high=100,size=(n_ics))
        print('random seeds are: ' ,random_seed)
    print('generating pendulum data, pendulum type: cart pendulum')
    'z[0]-theta z[1]-theta_dot'
    t = np.arange(0, 50, .02)
    '500 time steps'
    i = 0
    X, Xdot = [], []
    min_angle_limit = 1
    #shape of X and Xdot: (50000,4)
    #structure of X:q1,q2,q1_t,q2_t
    ##structure of Xdot:q1_t,q2_t,q1_tt,q2_tt
    while (i < n_ics):
        if params['specific_random_seed'] == True:
            np.random.seed(random_seed[i])
        if np.random.rand() > 0.5:
            theta = np.random.uniform(-np.pi/2, -min_angle_limit)
        else:
            theta = np.random.uniform(min_angle_limit, np.pi/2)
        if params['specific_random_seed'] == True:
            np.random.seed(random_seed[i])
        thetadot = np.random.uniform(0,0)
        y0=np.array([0, theta, 0, thetadot])
        cartPendulum = cartPendulum_wrapper(params)
        x,xdot = generate_data(cartPendulum,t,y0)
        X.append(x)
        Xdot.append(xdot)
        i += 1
    X = np.vstack(X)
    Xdot = np.vstack(Xdot)
    if params['adding_noise'] == True :
        if params['noise_type'] == 'angle_noise':
            print('Adding noise to the pendulum data')
            print('noise_type: angle noise')
            mu,sigma = 0,params['noiselevel']
            noise = np.random.normal(mu,sigma,X.shape[0])
            X_noise = np.zeros(X.shape)
            Xdot_noise = np.zeros(Xdot.shape)
            for i in range(X.shape[1]):
                X_noise[:,i] = X[:,i]+noise
                Xdot_noise[:,i] = Xdot[:,i]+noise
            X = X_noise
            Xdot = Xdot_noise
    return t,X,Xdot



#code for plotting the cart-pendulum to movie
params = {}
params['M'] = 1
params['m'] = 1
params['R'] = 1
params['k'] = 1
params['d'] = 0.1
params['b'] = 0.1
params['adding_noise'] = False
params['specific_random_seed'] = False
# Constants
length_of_pendulum =  params['R']  
data = get_pendulum_data(1,params)
data = data['z']
# Function to update the plot for each frame
def update(frame):
    plt.cla()  # Clear the current axes
    cart_pos = data[frame, 0]
    pendulum_angle = data[frame, 1]
    pendulum_x = cart_pos + length_of_pendulum * np.sin(pendulum_angle)
    pendulum_y = -length_of_pendulum * np.cos(pendulum_angle)

    # Draw cart
    plt.plot(cart_pos, 0, 'ks', markersize=12)  # 'ks' for black square

    # Draw pendulum
    plt.plot([cart_pos, pendulum_x], [0, pendulum_y], 'r-')  # Red line for pendulum
    plt.plot(pendulum_x, pendulum_y, 'ro')  # Red dot for pendulum bob

    # Setting plot limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal', adjustable='box')

# Create an animation
fig, ax = plt.subplots()
frames = len(data)  # Total number of frames in the animation
ani = FuncAnimation(fig, update, frames=frames, interval=20)

# Convert to a movie file using MoviePy
def make_frame(t):
    update(int(t * 50))  # Assuming 50 fps
    fig.canvas.draw()
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

animation = VideoClip(make_frame, duration=frames / 50)
animation.write_videofile("cart_pendulum_animation.mp4", fps=50)

print("Movie created successfully!")
