import numpy as np
import sys 


from sympy import symbols, simplify, derive_by_array, cos, sin, sympify
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
from adaPGM import adaptive_primal_dual, NormL1, Zero, OurRule
import sympy
import torch
import HLsearch as HL
import matplotlib.pyplot as plt
import math


import time

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])

def doublePendulum2_wrapper(params):
    def doublePendulum2(t, y):
        l1, l2, m1, m2, b1, b2, tau0, omega1, omega2, phi,g = params['L1'], params['L2'], params['m1'], params['m2'], params['b1'], params['b2'], params['tau0'], params['omega1'], params['omega2'], params['phi'], params['g']

        q1,q2,q1_t,q2_t = y

        q1_2t = (m2*l1*q1_t**2*np.sin(2*(q1-q2))+2*m2*l2*q2_t**2*np.sin(q1-q2)+2*g*m2*np.cos(q2)*np.sin(q1-q2)+2*g*m1*np.sin(q1)+(2*b1*q1_t-2*tau0*np.cos(omega1*t)-2*np.cos(q1-q2)*(b2*q2_t-tau0*np.cos(omega2*t))))/(-2*l1*(m1+m2*(np.sin(q1-q2)**2)))

        q2_2t = (m2*l2*q2_t**2*np.sin(2*(q1-q2))+2*(m1+m2)*l1*q1_t**2*np.sin(q1-q2)+2*(m1+m2)*g*np.cos(q1)*np.sin(q1-q2)+2*(b1*q1_t-tau0*np.cos(omega1*t))*np.cos(q1-q2)-(2*(m1+m2)/m2)*(b2*q2_t-tau0*np.cos(omega2*t)))/(2*l2*(m1+m2*(np.sin(q1-q2)**2)))


        return q1_t, q2_t, q1_2t, q2_2t
    return doublePendulum2

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

#ADAM optimizer
def SR_loop(Tau, coef, prevcoef, d_coef, RHS, Dissip, xdot, bs, epoch, lr, lam,beta1=0.9,beta2=0.999,eps=1e-8,D_CAL=False,device='cuda:0'):
    Zeta, Eta, Delta = RHS
    predcoef = coef.clone().detach().to(device).requires_grad_(True)
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]
    t = coef.shape[0]
    weight = coef.clone().detach().to(device).requires_grad_(True)
    if D_CAL:
        weight = torch.cat((weight, d_coef.clone().detach().to(device).requires_grad_(True)),0)

    # Initialize moving averages for Adam
    m = torch.zeros_like(weight)
    v = torch.zeros_like(weight)
    for j in range(epoch):
        for i in range(tl//bs):
            #computing acceleration with momentum
            what = weight.requires_grad_(True).clone().detach().to(device).requires_grad_(True)

            

            #Computing loss
            zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
            eta = Eta[:,:,:,i*bs:(i+1)*bs]
            delta = Delta[:,:,i*bs:(i+1)*bs]
            dissip = Dissip[:,:,i*bs:(i+1)*bs]
            tau = Tau[:,i*bs:(i+1)*bs]
            x_t = xdot[i*bs:(i+1)*bs,:]
            #forward
            pred, weight = ELDPforward(weight[:t], weight[t:], zeta, eta, delta,dissip, x_t, device,D_CAL)
            targ = tau 
            lossval = loss(pred, targ)
            l1_norm = torch.norm(weight[:t], 1)
            lossval = lossval + lam * l1_norm 
            #Backpropagation
            lossval.backward()
            with torch.no_grad():
                # Update moving averages
                m = beta1 * m + (1 - beta1) * weight.grad
                v = beta2 * v + (1 - beta2) * weight.grad ** 2
                # Compute bias-corrected moving averages
                m_hat = m / (1 - beta1 ** (i + 1))
                v_hat = v / (1 - beta2 ** (i + 1))
                # Update parameters
                weight = weight - lr * m_hat / (torch.sqrt(v_hat) + eps)
                #reset gradient
                weight.grad = None
            vhat = weight[:t]
            if D_CAL:
                dhat = weight[t:]

            loss_list.append(lossval.item())
        if j % 10 == 0:
            if display:
                print("Epoch : ", j, "/", epoch)
                print("Average loss : " , torch.tensor(loss_list).mean().item())
    return vhat, dhat, torch.tensor(loss_list).mean().item()

#PGD optimizer


def adaPGM(Tau, coef, RHS, Dissip, xdot, bs, lam):
    
    loss_list = []
    tl = xdot.shape[0]
    n = coef.shape[0]
    Zeta, Eta, Delta = RHS
    if(torch.is_tensor(xdot)==False):
        xdot = torch.from_numpy(xdot).to(device).float()

    class LinearLeastSquares():
        def __init__(self, A = 0, b = 0):
            self.A = A
            self.b = b

    
    zeta = Zeta
    eta = Eta
    delta = Delta
    tau = Tau
    x_t = xdot

    A = candidate_forward(zeta,eta,delta,x_t,device)
    A = A.reshape(-1, 79)
    tau = tau.reshape(-1)
    tau = tau.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    gam_init = 1 / np.linalg.norm(A,ord=2)**2
    f = LinearLeastSquares(A,tau)
    g = NormL1(lambda_=lam)
    coef, loss = adaptive_primal_dual(np.zeros(n), np.zeros(n), f, g, h = Zero(), A = 0, rule = OurRule(gamma = gam_init), tol = 1e-5, max_it = 3000)
    coef = torch.tensor(coef,device=device).float()
    #regularize the biggest coefficient to 10
    idx = torch.argmax(torch.abs(coef))
    coef = coef / coef[idx] * 10
    return coef, None, None, loss




    if(opt_mode=="ADAM"):
        return SR_loop(Tau, coef, prevcoef,d_coef, RHS, Dissip, xdot, bs, lr, lam)
    elif(opt_mode=="PGD"):
        return PGD_loop(Tau, coef, prevcoef,d_coef,RHS, Dissip, xdot, bs, lr, lam)
    elif(opt_mode=="adaPGM"):
        return adaPGM(Tau, coef, RHS, Dissip, xdot, bs, lam)
    else:
        print("Invalid opt_mode")
        return None


def main(param=None,device='cuda:6',opt_mode='PGD',num_sample=100,noiselevel=0,Epoch=100,Epoch0=200,lr=1e-7,lr_step=0,lam0=1,lam=0.1,batch_size=128,threshold_d=1e-6,tol=1e-5,display=True):
#default setting, works well for most cases
# def main(param=None,device='cuda:0',opt_mode='PGD',num_sample=100,noiselevel=0,Epoch=100,Epoch0=100,lr=4e-6,lr_step=1e-6,lam0=0.8,lam=0.1,batch_size=128,threshold_d=0):
#optuna best setting
# def main(param=None,device='cuda:0',opt_mode='PGD',num_sample=73,noiselevel=0,Epoch=231,Epoch0=348,lr=1.1e-6,lr_step=6e-6,lam0=0.8,lam=0.248,batch_size=256):
# device = 'cuda:7'
    if param is None:
        param = {}
        param['L1'] = 1
        param['L2'] = 1
        param['m1'] = 1
        param['m2'] = 1
        param['b1'] = 0.5
        param['b2'] = 0.5
        param['tau0'] = 0.2
        param['omega1'] = 0.5
        param['omega2'] = 0.3
        param['phi'] = 0
        param['g'] = 9.81
# The gravitational acceleration (m.s-2).
    
    doublePendulum = doublePendulum2_wrapper(param)
    #Saving Directory
    rootdir = "../Double Pendulum/Data/"
    create_data = True
    training = True
    save = False

    if(create_data):
        if display:
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
        #genrate sinodusal input using the omega and tau0
        Tau = np.array([param['tau0']*np.cos(param['omega1']*t),param['tau0']*np.cos(param['omega2']*t)])
        Tau = torch.tensor(Tau,device=device).float()
        #duplicate the input to match the size of the data
        Tau_temp = Tau
        for i in range(num_sample-1):
            Tau_temp = torch.cat((Tau_temp, Tau), dim=1)
        Tau = Tau_temp
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
    if display:
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
    # d_expr = ['x0_t**2','x1_t**2','x0_t','x1_t']
    d_expr = ['x0_t**2','x1_t**2']
    # expr = ['cos(x0)','cos(x1)','x0_t*x1_t*cos(x0)*cos(x1)','x0_t*x1_t*sin(x0)*sin(x1)','x0_t**2','x1_t**2']


    #Creating library tensor
    Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=True)


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
    idx = np.delete(idx,[i2,i3,i7,i8,i9,i10,i11,i12,i13,i14])

    expr = np.delete(expr,[i2,i3,i7,i8,i9,i10,i11,i12,i13,i14])

    #non-penalty index from prev knowledge
    i4 = np.where(expr == 'x1_t**2')[0][0]
    i5 = np.where(expr == 'cos(x0)')[0][0]
    i6 = np.where(expr == 'cos(x1)')[0][0]
    nonpenaltyidx = [i4, i5, i6]
    # nonpenaltyidx = []

    expr = expr.tolist()



    Zeta = Zeta[:,:,idx,:]
    Eta = Eta[:,:,idx,:]
    Delta = Delta[:,idx,:]



    #Moving to Cuda


    Zeta = Zeta.to(device)
    Eta = Eta.to(device)
    Delta = Delta.to(device)
    Dissip = Dissip.to(device)




    xi_L = torch.ones(len(expr), device=device).data.uniform_(-20,20)
    prevxi_L = xi_L.clone().detach()
    xi_d = torch.ones(len(d_expr), device=device)

    ## ADMM optimizer

    def ADMM_loop(Tau, coef, prevcoef, d_coef, RHS, Dissip, xdot, bs, lr, lam, momentum=True, D_CAL=False, device='cuda:0'):
        loss_list = []
        tl = xdot.shape[0]
        n = xdot.shape[1]
        Zeta, Eta, Delta = RHS
        if not torch.is_tensor(xdot):
            xdot = torch.from_numpy(xdot).to(device).float()
        
        v = coef.clone().detach().requires_grad_(True)
        # d = d_coef.clone().detach().requires_grad_(True)
        d = torch.tensor([-0.25,-0.25],device=device).float().requires_grad_(True)
        if D_CAL:
            weight = torch.cat((v,d),0).clone().detach().requires_grad_(True)
        else:
            weight = v.clone().detach().requires_grad_(True)
        # ADMM specific variables
        u = torch.zeros_like(weight)  # Dual update variable
        z = weight.clone()  # Auxiliary variable similar to v
        
        for i in range(tl // bs):
            zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
            eta = Eta[:,:,:,i*bs:(i+1)*bs]
            delta = Delta[:,:,i*bs:(i+1)*bs]
            dissip = Dissip[:,:,i*bs:(i+1)*bs]
            tau = Tau[:,i*bs:(i+1)*bs]
            x_t = xdot[i*bs:(i+1)*bs,:]

            for admm_iter in range(10):  # Perform ADMM iterations
                # Update v (minimize the loss with respect to v holding z and u fixed) 
                weight.requires_grad_(True)
                pred = ELDPforward_com(weight + u, zeta, eta, delta, dissip, x_t, device, D_CAL)
                loss = torch.sum((pred - tau)**2) + lam * torch.norm(weight[:len(coef)] - z[:len(coef)] + u[:len(coef)], p=1)
                loss.backward()
                with torch.no_grad():
                    weight = weight - lr * weight.grad
                    weight.grad = None
                
                # Update z (proximal update, typically soft thresholding in case of L1 norm)
                z_old = z.clone()
                with torch.no_grad():
                    z = soft_threshold(weight+u,lam)
                
                # Update u (dual ascent step)
                with torch.no_grad():
                    u += weight - z
                
                # Check convergence (optional, could implement based on threshold of changes in z)
                if torch.norm(z - z_old) < 1e-3 :
                    break
            loss_list.append(loss.item())
        v = z[:len(expr)]
        if D_CAL:
            d = z[len(expr):]
        if display:
            print("Average loss:", torch.tensor(loss_list).mean().item())
        return v, prevcoef, d, torch.tensor(loss_list).mean().item()

    def soft_threshold(x, kappa):
        return torch.sign(x) * torch.clamp(torch.abs(x) - kappa, min=0)




    i = 0
    temp = 1000
    RHS = [Zeta, Eta, Delta]

    while(i<=Epoch0):
        if display:
            print("\n")
            print("Stage 1")
            print("Epoch "+str(i) + "/" + str(Epoch0))
            print("Learning rate : ", lr)
        xi_L, prevxi_L, xi_d, lossitem= ADMM_loop(Tau, xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,batch_size,lr=lr,lam=lam0,D_CAL=True,device=device)
        temp = lossitem
        i+=1
        if math.isnan(temp):
            return xi_L,100


    ## Thresholding
    threshold = 0.01
    surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
    xi_d = xi_d.clone().detach().requires_grad_(True)
    prevxi_L = xi_L.clone().detach()

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-3)
    if display:
        if len(xi_L) < 20:
            print("Result stage 1: ", simplify(L))
        else:
            print("Result stage 1: ", L)

    last_ten_loss = []
    converged = False
    quiting = 0
    ## Next round selection ##
    for stage in range(100):

        #Redefine computation after thresholding
        
        Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=False)


        expr = np.array(expr)
        i1 = np.where(expr == 'x0_t**2')[0]
        i4 = np.where(expr == 'x1_t**2')[0][0]
        i5 = np.where(expr == 'cos(x0)')[0][0]
        i6 = np.where(expr == 'cos(x1)')[0][0]
        

        nonpenaltyidx = [i4,i5,i6]

        expr = expr.tolist()

        # nonpenaltyidx = []

        Zeta = Zeta.to(device)
        Eta = Eta.to(device)
        Delta = Delta.to(device)
        Dissip = Dissip.to(device)


      
        i = 0
        
        if(len(xi_L)+len(xi_d) < 8):
            lam = 0
            threshold = 1e-3
        # elif(len(xi_L) <= 8):
        #     lam = 0
        else:
            threshold = 0.01
            lr += lr_step
        if quiting == 1:
            threshold_d = 0.1

        temp = 1000
        RHS = [Zeta, Eta, Delta]
        
        while(i<=Epoch):
            if display:
                print("\n")
                print("Stage " + str(stage+2))
                print("Epoch "+str(i) + "/" + str(Epoch))
                print("Learning rate : ", lr)
            xi_L, prevxi_L, xi_d, lossitem= ADMM_loop(Tau, xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,batch_size,lr=lr,lam=lam,D_CAL=True,device=device)
            i+=1
            #attend to loss list, if the size of the loss list is less than 10, append the loss value, else pop the first element and append the new loss value
            if len(last_ten_loss) < 10:
                last_ten_loss.append(lossitem)
            else:
                last_ten_loss.pop(0)
                last_ten_loss.append(lossitem)
            #calculate the changes in the loss value, if the all changes are less than a threshold, break the loop
            if len(last_ten_loss) == 10:
                if all(abs(last_ten_loss[i] - last_ten_loss[i+1]) < tol for i in range(len(last_ten_loss)-1)):
                    if display:
                        print("training is converged")
                        print("last ten loss values : ",    last_ten_loss)
                    converged = True
            if math.isnan(temp):
                return xi_L,100
            if(temp <= 1e-3):
                break
        
        
        ## Thresholding
        if stage < 2 or len(xi_L) > 12:
            
            #regularize the biggest coefficient to 20
            idx = torch.argmax(torch.abs(xi_L))
            xi_Ltemp = xi_L / xi_L[idx] * 20
            surv_index = ((torch.abs(xi_Ltemp) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
            expr = np.array(expr)[surv_index].tolist()

            xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
            xi_d = xi_d.clone().detach().requires_grad_(True)
            prevxi_L = xi_L.clone().detach()
            if display:
                print(xi_L)
            ## obtaining analytical model
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
            # print("Result stage " + str(stage+2) + ":" , simplify(L))
            if display:
                print("Result stage " + str(stage+2) + ":" , L)
                if len(xi_L) < 20:
                    print("simplified : ", simplify(L))
                print("Dissipation : ", simplify(D))
            #if the training is converged, run a extra round with strict threshold to remove uneccessary terms, then break the loop
            if converged:
                if quiting == 2:
                    break
                else:
                    quiting += 1

        else:
            if display:
                print("thresholding using the simplified expression")
        ## Thresholding
            ## obtaining analytical model
            #calculate the relative threshold
            scaler = 20 / torch.abs(xi_L).max().item()
            xi_L = xi_L * scaler
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
            L_simplified = simplify(L)
            x0, x1,x0_t,x1_t = symbols('x0 x1 x0_t x1_t')
            coeff_dict = L_simplified.as_coefficients_dict()
            if display:
                print(coeff_dict)
            scaler = coeff_dict['x0_t**2']
            relative_threshold = threshold * scaler
            #check the value of the coefficients, if the value is less than the relative threshold, remove the term
            filter_dict = {}
            for key in coeff_dict.keys():
                if abs(coeff_dict[key]) > relative_threshold:
                    filter_dict[key] = coeff_dict[key]
            xi_L_value = list(filter_dict.values())
            xi_L = torch.tensor(xi_L_value,device=device,dtype=torch.float32).requires_grad_(True)
            expr_temp = list(filter_dict.keys())
            expr =[]
            for x in expr_temp:
                expr.append('{}'.format(x))
            xi_d = xi_d.clone().detach().requires_grad_(True)
            prevxi_L = xi_L.clone().detach()
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            #perform thresholding on the dissipation term without simplification
            surv_index = ((torch.abs(xi_d) >= threshold_d)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
            d_expr = np.array(d_expr)[surv_index].tolist()
            xi_d =xi_d[surv_index].clone().detach().requires_grad_(True)
            D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)

        
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            if display:
                print("Result stage " + str(stage+2) + ":" , L)
                print("Dissipation : ", simplify(D))
            if converged:
                #if the training is converged, run a extra round with strict threshold to remove uneccessary terms, then break the loop
                if quiting == 2:
                    total_epoch = stage * Epoch + Epoch0
                    break
                else:
                    quiting += 1



    ## Adding known terms
    L = str(simplify(L)) 
    D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
    if display:
        print("\m")
        print("Obtained Lagrangian : ", L)
        print("Obtained Dissipation : ", simplify(D))
    #caluclate the relative error of the obtained coefficients
    #the real Lagrangian model is m1*l1**2*x0_t**2/2 + m2*(l1**2*x0_t**2/2 + l2**2*x1_t**2/2 + l1*l2*x0_t*x1_t*cos(x0)*cos(x1)+l1*l2*x0_t*x1_t*sin(x0)*sin(x1)) + (m1+m2)*g*l1*cos(x0) + m2*g*l2*cos(x1)

    m1,m2,l1,l2,g = param['m1'],param['m2'],param['L1'],param['L2'],param['g']
    # Define the symbols
    x0, x1, x0_t, x1_t = symbols('x0 x1 x0_t x1_t')

    # Define the real Lagrangian model
    L_real = m1*l1**2*x0_t**2/2 + m2*(l1**2*x0_t**2/2 + l2**2*x1_t**2/2 + l1*l2*x0_t*x1_t*cos(x0)*cos(x1)+l1*l2*x0_t*x1_t*sin(x0)*sin(x1)) + (m1+m2)*g*l1*cos(x0) + m2*g*l2*cos(x1)

    # Simplify the real Lagrangian model if x0_t*x1_t*cos(x0 - x1) appears in the estimated candidates
    if 'x0_t*x1_t*cos(x0 - x1)' in expr:
        L_real_simplified = simplify(L_real)
    else:
        L_real_simplified = L_real

    # Get the real coefficients
    real_coeff_dict = L_real_simplified.as_coefficients_dict()
    # Create a dictionary of estimated coefficients
    estimated_coeff_dict = filter_dict
    

    #scale the x0_t**2 and use that scaler to scale the other coefficients
    scale = 1/estimated_coeff_dict[x0_t**2]

    for key in estimated_coeff_dict.keys():
        estimated_coeff_dict[key] = estimated_coeff_dict[key]*scale
    if display:
        print(estimated_coeff_dict)
    # Ensure that the real and estimated coefficients are in the same order
    real_coeff_values = []
    estimated_coeff_values = []
    for term in real_coeff_dict.keys():
        real_coeff_values.append(real_coeff_dict[term])
        # Use get method with default value 0 to avoid KeyError
        estimated_coeff_values.append(estimated_coeff_dict.get(str(term), 0))

    # Calculate the relative error
    # Initialize the sum of relative errors
    sum_relative_errors = 0

    # Calculate the relative error for each coefficient
    for cand in estimated_coeff_dict.keys():
        #check if the term is in the real coefficients
        if cand in real_coeff_dict.keys():
            real_coeff = real_coeff_dict[cand]
            estimated_coeff = estimated_coeff_dict[cand]
            relative_error = abs(real_coeff - estimated_coeff) / abs(real_coeff)
            sum_relative_errors += relative_error
        else:
            sum_relative_errors += 1

    # Print the relative errors
    if display:
        print("The relative errors are:", sum_relative_errors)
    





    if(save==True):
        #Saving Equation in string
        text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
        text_file.write(L)
        text_file.close()
    return estimated_coeff_dict, sum_relative_errors,total_epoch
if __name__ == "__main__":
    main()



