# %%
import numpy as np
import sys 


from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy_sp import *
from sympy.physics.mechanics import *
from sympy import *
from Data_generator_py import image_process
import sympy
import torch
import sys
import HLsearch as HL
import example_pendulum_cart_pendulum as example_pendulum
import time
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# %%
#set the random seed for reproducibility
seed_value = random.randint(0, 2**32 - 1)  # This generates a random integer in the range [0, 2^32 - 1]
#seed_value = 3489499403
#seed_value = 4027751856
# Set the seed for numpy
np.random.seed(seed_value)

# Set the seed for PyTorch
torch.manual_seed(seed_value)

# Also seed for cuda if you are using GPU computations
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Seed value: {seed_value}")


# %%
save = False
#set the environment for deciding the path to save the files
environment = "server"
sample_size = 30
device = 'cuda:4'

# %%
sys.path.append(r'../../../HLsearch/')
#set the parameters for the double pendulum
params = {}
params['adding_noise'] = False
params['noise_type'] = 'angle_noise'
params['noiselevel'] = 2e-2
params['changing_length'] = False
params['specific_random_seed'] = False
params['c'] = float(0.18)
params['g'] = float(9.81)
params['L1'] = float(1.5)
params['L2'] = float(1.5)
params['m1'] = float(1)
params['m2'] = float(1)
params['b1'] = float(0.05)
params['b2'] = float(0.05)
if environment == 'laptop':
    root_dir =R'C:\Users\87106\OneDrive\sindy\progress'
elif environment == 'desktop':
    root_dir = R'E:\OneDrive\sindy\progress'
elif environment == 'server':
    root_dir = R'/mnt/ssd1/stilrmy/Autoencoder-conservtive_expression'
#just named as image_process, but actually it is a simple raw data pass through now
x,dx,ddx = image_process(sample_size,params)


# %%
#store the raw data in X and Xdot variables
X = []
Xdot = []
state_log = []
for i in range(len(x)):
    temp_list = np.hstack([x[i,:],dx[i,:]])
    X.append(temp_list)
    temp_list = np.hstack([dx[i,:],ddx[i,:]])
    Xdot.append(temp_list)
    temp_list = np.hstack([x[i,:],dx[i,:],ddx[i,:]])
    state_log.append(temp_list)
X = np.vstack(X)
Xdot = np.vstack(Xdot)
state_log = np.vstack(state_log)
print(Xdot.shape)
#change X and Xdot dtype to float32 to match the network
X = X.astype('float32')
Xdot = Xdot.astype('float32')
state_log = state_log.astype('float32')
# %%
#setting the states and states derivatives
states_dim = 4
states = ()
states_dot = ()
for i in range(states_dim):
    if(i<states_dim//2):
        states = states + (symbols('x{}'.format(i)),)
        states_dot = states_dot + (symbols('x{}_t'.format(i)),)
    else:
        states = states + (symbols('x{}_t'.format(i-states_dim//2)),)
        states_dot = states_dot + (symbols('x{}_tt'.format(i-states_dim//2)),)
print('states are:',states)
print('states derivatives are: ', states_dot)

# %%
#Turn from sympy to str
states_sym = states
states_dot_sym = states_dot
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)

# %%
def loss(pred, targ):
    loss = torch.mean((pred - targ)**2) 
    return loss 

# %%
def clip(w, alpha):
    clipped = torch.minimum(w,alpa)
    clipped = torch.maximum(clipped,-alpha)
    return clipped

# %%
def proxL1norm(w_hat, alpha):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w = w_hat - clip(w_hat,alpha)
    return w

# %%
def scad_penalty(beta, lambda_val, a):
    # Ensure we're working with torch tensors
    beta = beta.clone().detach()
    lambda_val = torch.tensor(lambda_val, dtype=torch.float32)
    a = torch.tensor(a, dtype=torch.float32)

    # Apply the SCAD penalty element-wise
    abs_beta = torch.abs(beta)
    condition1 = abs_beta <= lambda_val
    condition2 = (abs_beta > lambda_val) & (abs_beta <= a * lambda_val)
    condition3 = abs_beta > a * lambda_val

    part1 = lambda_val * abs_beta * condition1
    part2 = ((2 * a * lambda_val * abs_beta - beta.pow(2) - lambda_val.pow(2)) / (2 * (a - 1))) * condition2
    part3 = (lambda_val.pow(2) * (a + 1) / 2) * condition3

    scad_penalty = part1 + part2 + part3
    return scad_penalty.sum() 

# %% [markdown]
# Here we have two sparse regression algorithms, the Prox_loop is the one that Adam used in the previous research, and the SR_loop is a Adaptive Moment Estimation optimizer(also named Adam). The SR_loop is better but neither of them are the best solution for sparse regression.

def train_with_ADMM(model, data_loader, rho, lam, max_iter=100):
    # Initialize z and w as zero tensors with the same shape as the model's output
    z = torch.zeros(76).to(device)
    w = torch.zeros(76).to(device)
    d_coef = torch.tensor([0.025,0.025]).to(device).float()
    loss_sum = 0
    for iteration in range(max_iter):
        for state_batch, xdot_batch, delta_batch, zeta_batch, eta_batch, dissip_batch in data_loader:
            delta_batch = delta_batch.permute(1,2,0)
            zeta_batch = zeta_batch.permute(1,2,3,0).to(device)
            eta_batch = eta_batch.permute(1,2,3,0).to(device)
            dissip_batch = dissip_batch.permute(1,2,0).to(device)
            # Step 1: Update x by training the neural network
            state_batch = state_batch.to(device).flatten()
            xdot_batch = xdot_batch.to(device)
            x_pred = model(state_batch)

            loss = lagrangianforward(x_pred, d_coef, zeta_batch, eta_batch, delta_batch, dissip_batch, xdot_batch, device)  # Replace with your actual loss function
            penalty = rho / 2 * torch.norm(z - x_pred + w)**2
            total_loss =  loss.mean() + penalty
            loss_sum += total_loss
            # Here you would manually compute the gradient with respect to x
            # and perform the parameter update, effectively replacing the optimizer.step() call
            grad_x = torch.autograd.grad(total_loss, model.parameters(), retain_graph=True)
            with torch.no_grad():
                for param, grad in zip(model.parameters(), grad_x):
                    param -= lr * grad  # Replace lr with an appropriate learning rate

            # Step 2: Update z using the proximal operator for SCAD
            z = proxSCAD(x_pred + w, lam / rho, 3.7)

            # Step 3: Update w
            w += rho * (x_pred - z)


    return model, x_pred, loss_sum / (len(data_loader)*max_iter)

#another training loop that use ADAM optimizer and L1 norm as the penalty
def train_with_ADAM(model, data_loader, opt, rho, lam, max_iter=100):
    d_coef = torch.tensor([0.025,0.025]).to(device).float()
    loss_sum = 0
    for iteration in range(max_iter):
        for state_batch, xdot_batch, delta_batch, zeta_batch, eta_batch, dissip_batch in data_loader:
            delta_batch = delta_batch.permute(1,2,0)
            zeta_batch = zeta_batch.permute(1,2,3,0).to(device)
            eta_batch = eta_batch.permute(1,2,3,0).to(device)
            dissip_batch = dissip_batch.permute(1,2,0).to(device)
            # Step 1: Update x by training the neural network
            state_batch = state_batch.to(device).flatten()
            xdot_batch = xdot_batch.to(device)
            x_pred = model(state_batch)

            loss = lagrangianforward(x_pred, d_coef, zeta_batch, eta_batch, delta_batch, dissip_batch, xdot_batch, device)
            #L1 norm penalty
            penalty = lam * torch.norm(x_pred, p=1)
            total_loss =  loss.mean() + penalty
            loss_sum += total_loss
            # compute gradient and step the optimizer
            opt.zero_grad()
            total_loss.backward()
            opt.step()
    return model, x_pred, loss_sum / (len(data_loader)*max_iter)


# %%
#a sanity check to see if any of the right candiates are missing
def check_candidates(expr_temp):
    candidates = ['x0_t**2', 'x1_t**2', 'cos(x0)', 'cos(x1)', 'x0_t*x1_t*cos(x0)*cos(x1)', 'x0_t*x1_t*sin(x0)*sin(x1)']
    for candidate in candidates:
        if candidate not in expr_temp:
            return False
            print('candidate {} not in expr_temp'.format(candidate))
    return True

# %%

# Initialize variables to keep track of success and total trials
total_trials = 100  # or any number you'd like
successful_trials = 0

# Loop for each trial
for trial in range(total_trials):
    #set the random seed for reproducibility
    seed_value = random.randint(0, 2**32 - 1)  # This generates a random integer in the range [0, 2^32 - 1]
    #seed_value = 3489499403
    #seed_value = 4027751856
    # Set the seed for numpy
    np.random.seed(seed_value)

    # Set the seed for PyTorch
    torch.manual_seed(seed_value)

    # Also seed for cuda if you are using GPU computations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Seed value: {seed_value}")
    # ... [Your existing code for setting up the problem, like building function expressions, goes here]
    # build function expression for the library in str
    exprdummy = HL.buildFunctionExpressions(1,states_dim,states,use_sine=True)
    polynom = exprdummy[2:4]
    trig = exprdummy[4:]
    polynom = HL.buildFunctionExpressions(2,len(polynom),polynom)
    trig = HL.buildFunctionExpressions(2, len(trig),trig)
    product = []
    for p in polynom:
        for t in trig:
            product.append(p + '*' + t)
    expr = polynom + trig + product
    expr = np.array(expr)
    #check Adam's xLSINDy paper for why we need to delete some of the expressions
    i2 = np.where(expr == 'x0_t**2*cos(x0)**2')[0]
    i3 = np.where(expr == 'x0_t**2*cos(x1)**2')[0]
    i7 = np.where(expr == 'x1_t*cos(x0)**2')[0]
    i8 = np.where(expr == 'x1_t*cos(x1)**2')[0]
    i9 = np.where(expr == 'x1_t')[0]
    i10 = np.where(expr == 'x0_t*cos(x0)**2')[0]
    i11 = np.where(expr == 'x0_t*cos(x1)**2')[0]
    i12 = np.where(expr == 'x0_t')[0]
    i13 = np.where(expr == 'cos(x0)**2')[0]
    i14 = np.where(expr == 'cos(x1)**2')[0]
    i15 = np.where(expr == 'sin(x0)**2')[0]
    i16 = np.where(expr == 'sin(x1)**2')[0]
    i17 = np.where(expr == 'x0_t*x1_t')[0]
    idx = np.arange(0,len(expr))
    idx = np.delete(idx,[i2,i3,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17])

    expr = np.delete(expr,[i2,i3,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17])
    #expr = ['x0_t**2','x1_t**2','cos(x0)','cos(x1)','x0_t*x1_t*cos(x0)*cos(x1)','x0_t*x1_t*sin(x0)*sin(x1)']
    #library function expression for the dissipation function in str
    #d_expr = [ 'x0_t','x0_t**2','x0_t**3','x1_t','x1_t**2','x1_t**3','x0_t*x1_t']
    d_expr = ['x0_t**2','x1_t**2']

    #defing the NN
    class SINDyNN(nn.Module):
        def __init__(self, input_size, output_size, hidden_layer_size):
            super(SINDyNN, self).__init__()
            # Define the network layers
            self.fc1 = nn.Linear(input_size, hidden_layer_size)  # Input layer to hidden layer
            self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)  # Hidden layer to hidden layer
            self.fc3 = nn.Linear(hidden_layer_size, output_size)  # Hidden layer to output layer
            self.relu = nn.ReLU()  # Non-linear activation function

        def forward(self, x):
            x = self.relu(self.fc1(x))  # Pass input through first layer and apply activation
            x = self.relu(self.fc2(x))  # Pass through second layer and apply activation
            x = self.fc3(x)  # Pass through output layer
            return x
    #defining the network
    model = SINDyNN(input_size=2500*6, output_size=len(expr), hidden_layer_size=2000).to(device)
    #print the network
    print(model)
    # Loop for the training process (or whatever your existing loop structure is)
    #building the partial deriative of the library function expression and calculate with the raw data
    Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot,device,scaling=True)
    Eta = Eta.to(device)
    Zeta = Zeta.to(device)
    Delta = Delta.to(device)
    Dissip = Dissip.to(device)
    #coefficients for the Lagrangian
    mask = torch.ones(len(expr),device=device)
    xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10).requires_grad_(True)
    #xi_L = torch.ones(len(expr), device=device)
    #xi_L = xi_L.type(torch.FloatTensor)
    prevxi_L = xi_L.clone().detach().to(device).requires_grad_(True)
    # coefficients for the dissipation function
    d_mask = torch.ones(len(d_expr),device=device)
    xi_d = torch.ones(len(d_expr),device=device)*0
    prevxi_d = xi_d.clone().detach()
    threshold = 0.001
    threshold_d = 0.001
    num_candidates_removed = 0
    stage = 1
    lr = 1e-5
    rho = 1e-2

    class dataset(Dataset):
        def __init__(self, state, Xdot, delta, zeta, eta, dissip):
            # Assume delta, zeta, eta, and dissip are all tensors with the same size along the l dimension
            self.delta = delta
            self.zeta = zeta
            self.eta = eta
            self.dissip = dissip
            self.state = state
            self.Xdot = Xdot
        def __len__(self):
            # The dataset size is determined by the l dimension (time)
            return self.delta.size(-1)

        def __getitem__(self, idx):
            # Fetch the data slice across the l dimension
            delta_slice = self.delta[..., idx]
            zeta_slice = self.zeta[..., idx]
            eta_slice = self.eta[..., idx]
            dissip_slice = self.dissip[..., idx]
            state_slice = self.state[idx]
            Xdot_slice = self.Xdot[idx]
            # Return the data maintaining the original structure
            return state_slice, Xdot_slice, delta_slice, zeta_slice, eta_slice, dissip_slice
        
    # Create the dataset
    data = dataset(state_log, Xdot, Delta, Zeta, Eta, Dissip)
    # Create the data loader
    data_loader = DataLoader(data, batch_size=2500, shuffle=False)
    #check the shape of the data
    # Assuming 'model' is your neural network model instance
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)  # lr is the learning rate

    while True:

        # ... [Your existing training or optimization code goes here]
        #Redefine computation after thresholding
        Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot,device,scaling=True)
        Eta = Eta.to(device)
        Zeta = Zeta.to(device)
        Delta = Delta.to(device)
        Dissip = Dissip.to(device)


        # Dissip = Dissip.to(device)
        # print(expr)
        # print(d_expr)
        # print("Zeta:",Zeta)
        # print("Eta:",Eta)
        # print("Delta:",Delta)
        # print("Dissip:",Dissip)

        #Training
        Epoch = 200
        i = 1
        if lr < 0.0008:
            lr += lr*0.5
        if len(xi_L) <= 25:
            reset_threshold = 400
            threshold = 0.01
        else:
            reset_threshold = 120
        if(stage==1):
            lam = 0
        else:
            lam = 1e-3
        if len(xi_L) <= 50:
            lam = 5e-4
        if len(xi_L) <= 30:
            lam = 1e-4
        #set the lam to zero when mask is equal to 11
        d_training = False
        temp = 1000
        if len(xi_L) == 6:
            threshold = 0
            lam = 0
        lossitem = 0
        while(i<=Epoch):   
            #xi_L , xi_d, prevxi_L, prevxi_d, lossitem, q= SR_loop(xi_L,xi_d,prevxi_L,prevxi_d,Zeta,Eta,Delta,Dissip,Xdot,2500,lr,lam,d_training)
            #xi_L,prevxi_L,lossitem,q = Prox_loop(xi_L,xi_d,prevxi_L,Zeta,Eta,Delta,Dissip,Xdot,500,lr,lam,device)
            d_coef = torch.tensor([0.025,0.025]).to(device).float()
            for state_batch, xdot_batch, delta_batch, zeta_batch, eta_batch, dissip_batch in data_loader:
                optimizer.zero_grad()
                delta_batch = delta_batch.permute(1,2,0)
                zeta_batch = zeta_batch.permute(1,2,3,0).to(device)
                eta_batch = eta_batch.permute(1,2,3,0).to(device)
                dissip_batch = dissip_batch.permute(1,2,0).to(device)
                # Step 1: Update x by training the neural network
                state_batch = state_batch.to(device).flatten()
                xdot_batch = xdot_batch.to(device)
                xi_L = model(state_batch)
                loss = lagrangianforward(xi_L, d_coef, zeta_batch, eta_batch, delta_batch, dissip_batch, xdot_batch, device)
                # loss_SCAD = scad_penalty(xi_L, lam, 3.7)
                penalty = lam * torch.norm(xi_L, p=1)
                loss = (loss**2).mean() + penalty
                lossitem += loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #     Prox_loop(xi_L,xi_d,prevxi_L,Zeta,Eta,Delta,Dissip,Xdot,500,lr,lam,device)  # Your function call here
            # print(prof)
            if i %200 == 0:
                lossitem = lossitem / (sample_size*Epoch)
                print("\n")
                print("Stage ",stage)
                print("Epoch "+str(i) + "/" + str(Epoch))
                print("Learning rate : ", lr)
                print("Average loss : " , lossitem)
            temp = lossitem
            # if temp <= 5:
            #     lr = 1e-5
            # if temp <= 2:
            #     lr = 1e-5
            # if temp <= 0.05:
            #     lr = 1e-5

            i+=1
            
        surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
        expr_temp = np.array(expr)[surv_index].tolist()

        xi_L =xi_L[surv_index].clone().detach().to(device).requires_grad_(True)
        num_candidates_removed += len(prevxi_L) - len(xi_L)
        prevxi_L = xi_L.clone().detach()
        mask = torch.ones(len(expr_temp),device=device)

        if num_candidates_removed >= reset_threshold:
            xi_L = torch.ones(len(expr_temp), device=device).data.uniform_(5,7)
            prevxi_L = xi_L.clone().detach()
            num_candidates_removed = 0

        xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=4)
        xi_dcpu = np.around(xi_d.detach().cpu().numpy(),decimals=4)
        L = HL.generateExpression(xi_Lcpu,expr_temp)
        D = HL.generateExpression(xi_dcpu,d_expr)
        print("expression length:\t",len(xi_L))
        print("Result stage " + str(stage+2))
        print("removed candidates:", num_candidates_removed)
        print("sanity check: ",check_candidates(expr_temp))
        print("The current success rate is ",successful_trials / (trial + 1) * 100,"%")
        stage += 1
        # Check the conditions
        num_candidates = len(xi_L)  # Retrieve the current number of candidates

        # Call the check_candidates function and store its output
        check_result = check_candidates(expr_temp)  # Your existing check_candidates function call

        # Check for success criteria
        if num_candidates == 6 and check_result:
            print(f"Trial {trial + 1} is successful.")
            successful_trials += 1
            break  # Exit the training loop for this trial

        # Check for termination criteria based on check_candidates output
        if not check_result:
            print(f"Trial {trial + 1} failed.")
            print("The current success rate is ",successful_trials / (trial + 1) * 100,"%")
            break  # Exit the training loop for this trial
            #print the current sucess rate
            

# Compute and display the success rate
success_rate = (successful_trials / total_trials) * 100
print(f"The success rate is {{success_rate}}%.")



