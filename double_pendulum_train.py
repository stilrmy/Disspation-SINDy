import numpy as np
import sys 


from sympy import symbols, simplify, derive_by_array, cos, sin, sympify
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
import sympy
import torch
import HLsearch as HL
import matplotlib.pyplot as plt


import time

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])




param = {}
param['L1'] = 1
param['L2'] = 1
param['m1'] = 1
param['m2'] = 1
param['b1'] = 0.5
param['b2'] = 0.5
# The gravitational acceleration (m.s-2).
g = 9.81
tau = 0



def doublePendulum2_wrapper(params):
    def doublePendulum2(t, y):
        L1, L2, m1, m2, b1, b2 = params['L1'], params['L2'], params['m1'], params['m2'], params['b1'], params['b2']

        q1,q2,q1_t,q2_t = y

        q1_2t = -b1*L2*q1_t/(L1**2*L2*m1 - L1**2*L2*m2*np.cos(q1 - q2)**2 + L1**2*L2*m2) + b2*L1*q2_t*np.cos(q1 - q2)/(L1**2*L2*m1 - L1**2*L2*m2*np.cos(q1 - q2)**2 + L1**2*L2*m2) - g*L1*L2*m1*np.sin(q1)/(L1**2*L2*m1 - L1**2*L2*m2*np.cos(q1 - q2)**2 + L1**2*L2*m2) - g*L1*L2*m2*np.sin(q1)/(L1**2*L2*m1 - L1**2*L2*m2*np.cos(q1 - q2)**2 + L1**2*L2*m2) + g*L1*L2*m2*np.sin(q2)*np.cos(q1 - q2)/(L1**2*L2*m1 - L1**2*L2*m2*np.cos(q1 - q2)**2 + L1**2*L2*m2) - L1**2*L2*m2*q1_t**2*np.sin(q1 - q2)*np.cos(q1 - q2)/(L1**2*L2*m1 - L1**2*L2*m2*np.cos(q1 - q2)**2 + L1**2*L2*m2) - L1*L2**2*m2*q2_t**2*np.sin(q1 - q2)/(L1**2*L2*m1 - L1**2*L2*m2*np.cos(q1 - q2)**2 + L1**2*L2*m2)

        q2_2t =  b1*L2*m2*q1_t*np.cos(q1 - q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) - b2*L1*m1*q2_t/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) - b2*L1*m2*q2_t/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) + g*L1*L2*m1*m2*np.sin(q1)*np.cos(q1 - q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) - g*L1*L2*m1*m2*np.sin(q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) + g*L1*L2*m2**2*np.sin(q1)*np.cos(q1 - q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) - g*L1*L2*m2**2*np.sin(q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) + L1**2*L2*m1*m2*q1_t**2*np.sin(q1 - q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) + L1**2*L2*m2**2*q1_t**2*np.sin(q1 - q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2) + L1*L2**2*m2**2*q2_t**2*np.sin(q1 - q2)*np.cos(q1 - q2)/(L1*L2**2*m1*m2 - L1*L2**2*m2**2*np.cos(q1 - q2)**2 + L1*L2**2*m2**2)

        return q1_t, q2_t, q1_2t, q2_2t
    return doublePendulum2

doublePendulum = doublePendulum2_wrapper(param)



#Saving Directory
rootdir = "../Double Pendulum/Data/"

num_sample = 100
create_data = True
training = True
save = False
noiselevel = 6e-2


if(create_data):
    print("Creating Data")
    X, Xdot = [], []
    for i in range(num_sample):
        t = np.arange(0,5,0.01)
        theta1 = np.random.uniform(-np.pi, np.pi)
        thetadot = np.random.uniform(0,0)
        theta2 = np.random.uniform(-np.pi, np.pi)
        
        y0=np.array([theta1, theta2, thetadot, thetadot])
        x,xdot = generate_data(doublePendulum,t,y0)
        X.append(x)
        Xdot.append(xdot)
    X = np.vstack(X)
    Xdot = np.vstack(Xdot)
    if(save==True):
        np.save(rootdir + "X.npy", X)
        np.save(rootdir + "Xdot.npy",Xdot)
else:
    X = np.load(rootdir + "X.npy")
    Xdot = np.load(rootdir + "Xdot.npy")


#adding noise
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X.shape[0])
for i in range(X.shape[1]):
    X[:,i] = X[:,i]+noise
    Xdot[:,i] = Xdot[:,i]+noise


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


#Turn from sympy to str
states_sym = states
states_dot_sym = states_dot
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)


#build function expression for the library in str
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
print(len(expr))
exit()
d_expr = ['x0_t**2','x1_t**2']


#Creating library tensor
Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=True)


## separating known and unknown terms ##
expr = np.array(expr)
i1 = np.where(expr == 'x0_t**2')[0]

## Garbage terms ##

'''
Explanation :
x0_t, x1_t terms are not needed and will always satisfy EL's equation.
Since x0_t, x1_t are garbages, we want to avoid (x0_t*sin()**2 + x0_t*cos()**2), thus we remove
one of them, either  x0_t*sin()**2 or x0_t*cos()**2. 
Since the known term is x0_t**2, we also want to avoid the solution of (x0_t**2*sin()**2 + x0_t**2*cos()**2),
so we remove either one of x0_t**2*sin()**2 or x0_t**2*cos()**2.
'''

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

#Deleting unused terms 
idx = np.arange(0,len(expr))
idx = np.delete(idx,[i1,i2,i3,i7,i8,i9,i10,i11,i12,i13,i14])
known_expr = expr[i1].tolist()
expr = np.delete(expr,[i1,i2,i3,i7,i8,i9,i10,i11,i12,i13,i14])

#non-penalty index from prev knowledge
i4 = np.where(expr == 'x1_t**2')[0][0]
i5 = np.where(expr == 'cos(x0)')[0][0]
i6 = np.where(expr == 'cos(x1)')[0][0]
#nonpenaltyidx = [i4, i5, i6]
nonpenaltyidx = []

expr = expr.tolist()
Zeta_ = Zeta[:,:,i1,:].clone().detach()
Eta_ = Eta[:,:,i1,:].clone().detach()
Delta_ = Delta[:,i1,:].clone().detach()

Zeta = Zeta[:,:,idx,:]
Eta = Eta[:,:,idx,:]
Delta = Delta[:,idx,:]


#Moving to Cuda
device = 'cuda:0'

Zeta = Zeta.to(device)
Eta = Eta.to(device)
Delta = Delta.to(device)
Dissip = Dissip.to(device)

Zeta_ = Zeta_.to(device)
Eta_ = Eta_.to(device)
Delta_ = Delta_.to(device)


xi_L = torch.ones(len(expr), device=device).data.uniform_(-20,20)
prevxi_L = xi_L.clone().detach()
xi_d = torch.ones(len(d_expr), device=device).data.uniform_(-20,20)
c = torch.ones(len(known_expr), device=device)


def loss(pred, targ):
    loss = torch.mean((pred - targ)**2) 
    return loss 


def clip(w, alpha):
    clipped = torch.minimum(w,alpha)
    clipped = torch.maximum(clipped,-alpha)
    return clipped

def proxL1norm(w_hat, alpha, nonpenaltyidx):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w = w_hat - clip(w_hat,alpha)
    for idx in nonpenaltyidx:
        w[idx] = w_hat[idx]
    return w


def training_loop(c,coef, prevcoef,d_coef, RHS, LHS, Dissip, xdot, bs, lr, lam, momentum=True):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]

    Zeta_, Eta_, Delta_ = LHS
    Zeta, Eta, Delta = RHS

    if(torch.is_tensor(xdot)==False):
        xdot = torch.from_numpy(xdot).to(device).float()
    
    v = coef.clone().detach().requires_grad_(True)
    d = d_coef.clone().detach().requires_grad_(True)
    prev = v
    pre_d = d
    
    for i in range(tl//bs):
                
        #computing acceleration with momentum
        if(momentum==True):
            vhat = (v + ((i-1)/(i+2))*(v - prev)).clone().detach().requires_grad_(True)
            # dhat = (d + ((i-1)/(i+2))*(d - pre_d)).clone().detach().requires_grad_(True)
            dhat = d.requires_grad_(True).clone().detach().requires_grad_(True)
        else:
            vhat = v.requires_grad_(True).clone().detach().requires_grad_(True)
            dhat = d.requires_grad_(True).clone().detach().requires_grad_(True)
   
        prev = v

        #Computing loss
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]

        zeta_ = Zeta_[:,:,:,i*bs:(i+1)*bs]
        eta_ = Eta_[:,:,:,i*bs:(i+1)*bs]
        delta_ = Delta_[:,:,i*bs:(i+1)*bs]

        dissip = Dissip[:,:,i*bs:(i+1)*bs]
        
        x_t = xdot[i*bs:(i+1)*bs,:]



        #forward
        disp = DPforward(dhat,dissip,device)
        pred = -ELforward(vhat,zeta,eta,delta,x_t,device)
        targ = ELforward(c,zeta_,eta_,delta_,x_t,device) + disp
        lossval = loss(pred, targ)
        
        #Backpropagation
        lossval.backward()

        with torch.no_grad():
            v = vhat - lr*vhat.grad
            v = (proxL1norm(v,lr*lam,nonpenaltyidx))
            d = dhat - lr*dhat.grad
            # Manually zero the gradients after updating weights
            vhat.grad = None
            dhat.grad = None
        
        
    
        
        loss_list.append(lossval.item())
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return v, prevcoef,d, torch.tensor(loss_list).mean().item()


Epoch = 100
i = 0
lr = 5e-6
lam = 0.1
temp = 1000
RHS = [Zeta, Eta, Delta]
LHS = [Zeta_, Eta_, Delta_]
while(i<=Epoch):
    print("\n")
    print("Stage 1")
    print("Epoch "+str(i) + "/" + str(Epoch))
    print("Learning rate : ", lr)
    xi_L, prevxi_L, xi_d, lossitem= training_loop(c, xi_L,prevxi_L,xi_d,RHS,LHS,Dissip,Xdot,128,lr=lr,lam=lam)
    temp = lossitem
    i+=1


## Thresholding
threshold = 1e-2
surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
expr = np.array(expr)[surv_index].tolist()

xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
xi_d = xi_d.clone().detach().requires_grad_(True)
prevxi_L = xi_L.clone().detach()

## obtaining analytical model
xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-3)
print("Result stage 1: ", simplify(L))


## Next round selection ##
for stage in range(6):

    #Redefine computation after thresholding
    expr.append(known_expr[0])
    
    Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=False)


    expr = np.array(expr)
    i1 = np.where(expr == 'x0_t**2')[0]
    i4 = np.where(expr == 'x1_t**2')[0][0]
    i5 = np.where(expr == 'cos(x0)')[0][0]
    i6 = np.where(expr == 'cos(x1)')[0][0]
    idx = np.arange(0,len(expr))
    idx = np.delete(idx,i1)
    known_expr = expr[i1].tolist()
    expr = np.delete(expr,i1).tolist()
    # nonpenaltyidx = [i4,i5,i6]
    nonpenaltyidx = []

    Zeta_ = Zeta[:,:,i1,:].clone().detach()
    Eta_ = Eta[:,:,i1,:].clone().detach()
    Delta_ = Delta[:,i1,:].clone().detach()

    Zeta = Zeta[:,:,idx,:]
    Eta = Eta[:,:,idx,:]
    Delta = Delta[:,idx,:]

    Zeta = Zeta.to(device)
    Eta = Eta.to(device)
    Delta = Delta.to(device)
    Zeta_ = Zeta_.to(device)
    Eta_ = Eta_.to(device)
    Delta_ = Delta_.to(device)
    Dissip = Dissip.to(device)

    Epoch = 80
    i = 0
    lr += 2e-6
    if(stage==4):
        lam = 0
    else:
        lam = 0.1
    temp = 1000
    RHS = [Zeta, Eta, Delta]
    LHS = [Zeta_, Eta_, Delta_]
    while(i<=Epoch):
        print("\n")
        print("Stage " + str(stage+2))
        print("Epoch "+str(i) + "/" + str(Epoch))
        print("Learning rate : ", lr)
        xi_L, prevxi_L, xi_d, lossitem= training_loop(c, xi_L,prevxi_L,xi_d,RHS,LHS,Dissip,Xdot,128,lr=lr,lam=lam)
        i+=1
        if(temp <= 1e-3):
            break
    
    ## Thresholding
    if stage < 1:
        threshold = 1e-1
        surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
        expr = np.array(expr)[surv_index].tolist()

        xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
        xi_d = xi_d.clone().detach().requires_grad_(True)
        prevxi_L = xi_L.clone().detach()
        print(xi_L)
        ## obtaining analytical model
        xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
        L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
        D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
        print("Result stage " + str(stage+2) + ":" , simplify(L))
        print("Dissipation : ", simplify(D))
    else:
       ## Thresholding
        threshold = 1e-1
        ## obtaining analytical model
        xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
        L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
        D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
        L_simplified = simplify(L)
        x0, x1,x0_t,x1_t = symbols('x0 x1 x0_t x1_t')
        coeff_dict = L_simplified.as_coefficients_dict()
        filter_dict = {key: val for key, val in coeff_dict.items() if val >= threshold}
        xi_L_value = list(filter_dict.values())
        xi_L = torch.tensor(xi_L_value,device=device,dtype=torch.float32).requires_grad_(True)
        expr_temp = list(filter_dict.keys())
        expr =[]
        for x in expr_temp:
            expr.append('{}'.format(x))
        xi_d = xi_d.clone().detach().requires_grad_(True)
        prevxi_L = xi_L.clone().detach()
        xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
        L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
        print("Result stage " + str(stage+2) + ":" , L)
        print("Dissipation : ", simplify(D))


## Adding known terms
L = str(simplify(L)) + " + " + known_expr[0]
D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
print("\m")
print("Obtained Lagrangian : ", L)
print("Obtained Dissipation : ", simplify(D))

expr = expr + known_expr
xi_L = torch.cat((xi_L, c))
mask = torch.ones(len(xi_L),device=device)


if(save==True):
    #Saving Equation in string
    text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
    text_file.write(L)
    text_file.close()





