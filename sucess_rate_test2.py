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
import example_pendulum_double_pendulum as example_pendulum
import time
import random
import torch.nn as nn

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Seed value: {seed_value}")


# %%
save = False
#set the environment for deciding the path to save the files
environment = "server"
sample_size = 20
device = 'cuda:2'

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
for i in range(len(x)):
    temp_list = np.hstack([x[i,:],dx[i,:]])
    X.append(temp_list)
    temp_list = np.hstack([dx[i,:],ddx[i,:]])
    Xdot.append(temp_list)
X = np.vstack(X)
Xdot = np.vstack(Xdot)
print(Xdot.shape)
#change X and Xdot dtype to float32 to match the network
X = X.astype('float32')
Xdot = Xdot.astype('float32')

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

def proxSCAD(v, lam, a):
    """Direct Proximal operator for the SCAD penalty based on the paper's formula."""
    abs_v = torch.abs(v)
    
    condition1 = (abs_v <= 2*lam)
    condition2 = (abs_v > 2*lam) & (abs_v <= a * lam)
    condition3 = (abs_v > a * lam)
    
    prox_v = torch.where(condition1, torch.sign(v) * torch.clamp(torch.abs(v) - lam, min=0),
                         torch.where(condition2, ((a-1)*v-torch.sign(v)*a*lam)/(a-2),v))
    
    return prox_v


# %% [markdown]
# Here we have two sparse regression algorithms, the Prox_loop is the one that Adam used in the previous research, and the SR_loop is a Adaptive Moment Estimation optimizer(also named Adam). The SR_loop is better but neither of them are the best solution for sparse regression.

# %%
#sparse regression using the APG(Accelerated Proximal Gradient Methods)
def Prox_loop(coef,d_coef,prevcoef,Zeta,Eta,Delta,Dissip,xdot,bs,lr,lam,device):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]
    # v = coef.clone().detach().to(device).requires_grad_(True)
    # prev = prevcoef.clone().detach().to(device).requires_grad_(True)
    v = coef
    prev = prevcoef

    for i in range(tl//bs):
        vhat = (v + ((i - 1) / (i + 2)) * (v - prev)).clone().detach().requires_grad_(True)
        prev = v

        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]
        dissip = Dissip[:,:,i*bs:(i+1)*bs]
        
        x_t = torch.tensor(xdot[i*bs:(i+1)*bs,:]).to(device)
        d_coef = torch.tensor([0.025,0.025]).to(device).float()
        
        
        loss = lagrangianforward(vhat,d_coef,zeta,eta,delta,dissip,x_t,device)
        loss = torch.mean(loss**2)
        loss.backward()
        
        
        
        with torch.no_grad():
            v = vhat - lr * vhat.grad
            v = proxSCAD(v,lam,3.7)
            vhat.grad = None
        loss_list.append(loss)
        


    return v,prev,torch.tensor(loss_list).mean().item(),tl

#Sparse regression using the ADMM algorithm
def ADMM_Prox_loop(coef, d_coef, prevcoef, Zeta, Eta, Delta, Dissip, xdot, bs, lr, lam, rho, device, max_iter=100):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]
    
    x = coef
    z = torch.zeros_like(coef)
    w = torch.zeros_like(coef)
    
    for i in range(tl // bs):
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]
        dissip = Dissip[:,:,i*bs:(i+1)*bs]
        x_t = torch.tensor(xdot[i*bs:(i+1)*bs,:]).to(device)
        
        # Step 1: Update x
        # In this case, your lagrangianforward plays the role of P(x)
        x.requires_grad = True
        loss = lagrangianforward(x, d_coef, zeta, eta, delta, dissip, x_t, device)
        penalty = rho / 2 * torch.norm(z - x + w)**2
        total_loss = torch.mean(loss**2) + penalty
        total_loss.backward(retain_graph=True)
        with torch.no_grad():
            x -= lr * x.grad
            x.grad.zero_()
        
        # Step 2: Update z using the proximal operator
        z = proxSCAD(x + w, lam / rho, 3.7)
        
        # Step 3: Update w
        w += rho * (x - z)
        
        loss_list.append(total_loss.item())

    return x, torch.tensor(loss_list).mean().item(), tl


#Sparse regression using the ADAM algorithm
def SR_loop(coef,d_coef, prevcoef, predcoef, Zeta, Eta, Delta,Dissip, xdot, bs, lr, lam,d_training,beta1=0.9,beta2=0.999,eps=1e-8):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]
    #if(torch.is_tensor(xdot)==False):
        #xdot = torch.from_numpy(xdot).to(device).float()
    v = coef.clone().detach().to(device).requires_grad_(True)
    v_d = d_coef.clone().detach().to(device).requires_grad_(True)
    prev = prevcoef.clone().detach().to(device).requires_grad_(True)
    prev_d = predcoef.clone().detach().to(device).requires_grad_(True)
    # Initialize moving averages for Adam
    m_v = torch.zeros_like(v)
    m_d = torch.zeros_like(v_d)
    v_v = torch.zeros_like(v)
    v_d_ = torch.zeros_like(v_d)
    for i in range(tl//bs):
        #computing acceleration with momentum
        vhat = v.requires_grad_(True).clone().detach().to(device).requires_grad_(True)
        vdhat = v_d.requires_grad_(True).clone().detach().to(device).requires_grad_(True)
        prev = v
        prev_d = v_d
        #Computing loss
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]
        dissip = Dissip[:,:,i*bs:(i+1)*bs]
        x_t = torch.tensor(xdot[i*bs:(i+1)*bs,:]).to(device)
        vdhat_ = torch.tensor([0.025,0.025]).to(device).float()
        #forward
        lossval = lagrangianforward(vhat,vdhat_,zeta,eta,delta,dissip,x_t,device)
        #lossval = torch.mean(lossval[0,:]**2)+torch.mean(lossval[1,:]**2)
        lossval = torch.mean(lossval**2)
        l1_norm = torch.norm(vhat, 1)
        l1_d_norm = torch.norm(vdhat, 1)
        lossval = lossval + lam * l1_norm 
        #Backpropagation
        lossval.backward()
        #torch.nn.utils.clip_grad_norm_(vhat, max_norm=6)
        #torch.nn.utils.clip_grad_norm_(vdhat, max_norm=6)
        with torch.no_grad():
            # Update moving averages
            m_v = beta1 * m_v + (1 - beta1) * vhat.grad
            v_v = beta2 * v_v + (1 - beta2) * (vhat.grad ** 2)
            # m_d = beta1 * m_d + (1 - beta1) * vdhat.grad
            #v_d_ = beta2 * v_d_ + (1 - beta2) * (vdhat.grad ** 2)
            # Compute bias-corrected moving averages
            m_v_hat = m_v / (1 - beta1 ** (i + 1))
            v_v_hat = v_v / (1 - beta2 ** (i + 1))
            m_d_hat = m_d / (1 - beta1 ** (i + 1))
            v_d_hat = v_d_ / (1 - beta2 ** (i + 1))
            # Update parameters
            v = vhat - lr * m_v_hat / (torch.sqrt(v_v_hat) + eps)
            #v_d = vdhat - lr * m_d_hat / (torch.sqrt(v_d_hat) + eps)
            #reset gradient
            vhat.grad.zero_()
            #vdhat.grad.zero_()

        loss_list.append(lossval.item())
    return v, vdhat_, prev, prev_d, torch.tensor(loss_list).mean().item(),tl


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
    rho = 1e-3
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
        # if lr < 0.0008:
        #     lr += lr*0.4
        if len(xi_L) <= 25:
            reset_threshold = 400
            threshold = 0.01
        else:
            reset_threshold = 120
        if(stage==1):
            lam = 0
        else:
            lam = 1e-3
        if len(xi_L) <= 30:
            lam = 1e-4
        #set the lam to zero when mask is equal to 11
        d_training = False
        temp = 1000
        if len(xi_L) == 6:
            threshold = 0
            lam = 0
        while(i<=Epoch):   
            #xi_L , xi_d, prevxi_L, prevxi_d, lossitem, q= SR_loop(xi_L,xi_d,prevxi_L,prevxi_d,Zeta,Eta,Delta,Dissip,Xdot,2500,lr,lam,d_training)
            xi_L,prevxi_L,lossitem,q = Prox_loop(xi_L,xi_d,prevxi_L,Zeta,Eta,Delta,Dissip,Xdot,500,lr,lam,device)
            #xi_L,lossitem,q = ADMM_Prox_loop(xi_L,xi_d,prevxi_L,Zeta,Eta,Delta,Dissip,Xdot,500,lr,lam,rho,device)
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #     Prox_loop(xi_L,xi_d,prevxi_L,Zeta,Eta,Delta,Dissip,Xdot,500,lr,lam,device)  # Your function call here
            # print(prof)

            if i %200 == 0:
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
        expr = np.array(expr)[surv_index].tolist()

        xi_L =xi_L[surv_index].clone().detach().to(device).requires_grad_(True)
        num_candidates_removed += len(prevxi_L) - len(xi_L)
        prevxi_L = xi_L.clone().detach()
        mask = torch.ones(len(expr),device=device)

        if num_candidates_removed >= reset_threshold:
            xi_L = torch.ones(len(expr), device=device).data.uniform_(5,7)
            prevxi_L = xi_L.clone().detach()
            num_candidates_removed = 0

        xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=4)
        xi_dcpu = np.around(xi_d.detach().cpu().numpy(),decimals=4)
        L = HL.generateExpression(xi_Lcpu,expr)
        D = HL.generateExpression(xi_dcpu,d_expr)
        print("expression length:\t",len(xi_L))
        print("Result stage " + str(stage+2))
        print("removed candidates:", num_candidates_removed)
        print("sanity check: ",check_candidates(expr))
        print("The current success rate is ",successful_trials / (trial + 1) * 100,"%")
        stage += 1
        # Check the conditions
        num_candidates = len(xi_L)  # Retrieve the current number of candidates

        # Call the check_candidates function and store its output
        check_result = check_candidates(expr)  # Your existing check_candidates function call

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



