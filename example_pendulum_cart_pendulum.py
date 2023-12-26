import numpy as np
import random
import os
import csv
from scipy.integrate import odeint
from PIL import Image
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Pendulum rod lengths (m), bob masses (kg).

# The gravitational acceleration (m.s-2).
g = 9.81
tau = 0
attenuation_rate = 100

density = 200 #density of the bar in the image
def get_pendulum_data(n_ics,params):
    # The gravitational acceleration (m.s-2).
    g = 9.81
    t,x,x2,X,Xdot = generate_pendulum_data(n_ics,params)

    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics*t.size, -1))
    data['x2'] = x2.reshape((n_ics*t.size, -1))
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


def plot(n_ics,params):
    t, x, x2, X, Xdot = generate_pendulum_data(n_ics,params)
    print(x.shape)
    imglist = []
    for i in range(500):
        image = Image.fromarray(x[i,:,:])
        print(image)
        imglist.append(image)
    imglist[0].save('save_name.gif', save_all=True, append_images=imglist, duration=0.1)
    imglist = []
    for i in range(500):
        image = Image.fromarray(x2[i,:,:])
        print(image)
        imglist.append(image)
    imglist[0].save('save_name2.gif', save_all=True, append_images=imglist, duration=0.1)
    return



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
    #shape of X and Xdot: (50000,4)
    #structure of X:q1,q2,q1_t,q2_t
    ##structure of Xdot:q1_t,q2_t,q1_tt,q2_tt
    while (i < n_ics):
        if params['specific_random_seed'] == True:
            np.random.seed(random_seed[i])
        theta = np.random.uniform(-np.pi/2, np.pi/2)
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
            x,x2 = pendulum_to_movie(X_noise,Xdot_noise,n_ics,params)
    x,x2 = pendulum_to_movie(X,Xdot,n_ics,params)
    return t,x,x2,X,Xdot


# code for visualization, it seperate the double pendulum into two single pendulums for NN processing to extract the states. And this part is abandoned so you prabably don't need to read it.
def generate_bar(theta1,theta2,L1,L2,params):
    bar = np.zeros([50,50])
    if params['sample_mode'] == 'bar':
        n = 240
        y1,y2 = np.meshgrid(np.linspace(-2.5,2.5,n),np.linspace(2.5,-2.5,n))
        LEN = np.linspace(0,L1,density)
        attenuation_rate0 = 400
        create_bar = lambda theta1,theta2,L,attenuation_rate:np.exp(-((y1-L*np.cos(theta1-np.pi/2))**2 + (y2-L*np.sin(theta1-np.pi/2))**2)*attenuation_rate0)
        for L in LEN:
            bar += create_bar(theta1,theta2,L,attenuation_rate0)*255
        create_bar_2 = lambda theta1,theta2,L1,L,attenuation_rate:np.exp(-((y1-L*np.cos(theta2-np.pi/2)-L1*np.cos(theta1-np.pi/2))**2 + (y2-L*np.sin(theta2-np.pi/2)-L1*np.sin(theta1-np.pi/2))**2)*attenuation_rate0)
        LEN = np.linspace(0,L2,density)
        for L in LEN:
            bar += create_bar_2(theta1,theta2,L1,L,attenuation_rate0)*255
    return bar


# code for visualization, it seperate the double pendulum into two single pendulums for NN processing to extract the states. And this part is abandoned so you prabably don't need to read it.
def pendulum_to_movie(X,Xdot,n_ics,params):
    n_samples = 2500
    n = 50
    y1,y2 = np.meshgrid(np.linspace(-2.5,2.5,n),np.linspace(2.5,-2.5,n))
    y3,y4 = np.meshgrid(np.linspace(-2.5,2.5,n),np.linspace(2.5,-2.5,n))



    x = np.zeros((n_ics*n_samples, n, n))
    x2 = np.zeros((n_ics*n_samples, n, n))
    center_dot = np.zeros([50,50])
    center_dot
    for i in range(X.shape[0]):
        theta = X[i,0]
        if params['changing_length'] == True:
            len = random.uniform(0.2,1)
        else:
            len = 1
        x[i, :, :] = create_image(X[i, 0], params['R'],attenuation_rate)*255
        x[i,X[i,1]-3:X[i,1]+3,25:28]=255
    
    return x


# def compare(params):
#     #generate data use double_pendulum and double_pendulum2 respectively with same initial values, and compare the results
#     t,x,x2,X,Xdot,X_,Xdot_ = generate_pendulum_data2(1,params)
#     #plot the results
#     plt.figure()
#     plt.subplot(2,2,1)
#     plt.plot(t,X[:,0],label='theta1')
#     plt.plot(t,X_[:,0],label='theta1_')
#     plt.legend()
#     plt.subplot(2,2,2)
#     plt.plot(t,X[:,1],label='theta2')
#     plt.plot(t,X_[:,1],label='theta2_')
#     plt.legend()
#     plt.subplot(2,2,3)
#     plt.plot(t,X[:,2],label='theta1_t')
#     plt.plot(t,X_[:,2],label='theta1_t_')
#     plt.legend()
#     plt.subplot(2,2,4)
#     plt.plot(t,X[:,3],label='theta2_t')
#     plt.plot(t,X_[:,3],label='theta2_t_')
#     plt.legend()
#     plt.show()
#     #save the figure
#     plt.savefig('compare.png')

# params = {}
# params['sample_mode'] = 'dot'
# params['adding_noise'] = False
# params['changing_length'] = False
# params['specific_random_seed'] = True
# params['random_seed'] = 3614

# compare(params)


