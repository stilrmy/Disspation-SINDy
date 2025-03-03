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

def cartPendulum2_wrapper(params):
    def cartPendulum2(t, y):
        l, M, m, b1, b2, tau, omega, phi,g = params['L'], params['M'], params['m'], params['b1'], params['b2'], params['tau'],params['omega'] , params['phi'], params['g']

        theta,x,thetadot,xdot = y
        F = tau*np.cos(omega*t)
        num = (F + m * g * np.sin(theta) * np.cos(theta) + (b2 / l) * thetadot * np.cos(theta) + m * l * thetadot**2 * np.sin(theta) - b1 * xdot)
        denom = (M + m - m * l * np.cos(theta)**2)
        xdotdot = num / denom
    
        # Compute theta_ddot
        thetadotdot = (-g / l * np.sin(theta) - xdotdot * np.cos(theta) - (b2 / (m * l**2)) * thetadot)
        # xdotdot = (tau*np.cos(omega*t)+m*np.sin(theta)*(l*thetadot**2+g*np.cos(theta))-b1*xdot)/(M+m*(np.sin(theta)**2))

        # thetadotdot = (-tau*np.cos(omega*t)*np.cos(theta) - m*l*thetadot**2*np.sin(theta)*np.cos(theta) - (M+m)*g*np.sin(theta)-b2*thetadot)/(l*(M+m*(np.sin(theta)**2)))


        return thetadot,xdot,thetadotdot,xdotdot
    return cartPendulum2

def cartpole(t,y,f=0.0):
    mc,mp,g = 1, 0.5, 9.81
    l = 1
    theta,x,thetadot,xdot = y
    f = 0.1 * np.cos(t)
    xdotdot = (f+mp*np.sin(theta)*(l*thetadot**2+g*np.cos(theta)))/(mc+mp*np.sin(theta)**2)
    thetadotdot = (-f*np.cos(theta)-mp*l*thetadot**2*np.cos(theta)*np.sin(theta)-(mc+mp)*g*np.sin(theta))/(l*(mc+mp*np.sin(theta)**2))
    return thetadot,xdot,thetadotdot,xdotdot

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


def sanity_check(expr):
    real_candidates = ['x1_t**2', 'cos(x0)']
    found = {cand: False for cand in real_candidates}  # Dictionary to track found items
    
    for cand in real_candidates:
        for item in expr:
            if cand in item:
                found[cand] = True  # Mark as found
    
    # Check if any candidate was not found
    for cand, is_found in found.items():
        if not is_found:
            print("Lacking of term:", cand)
            return False
    return True


def main(param=None,device='cuda:4',opt_mode='PGD',num_sample=100,noiselevel=0,Epoch=100,Epoch0=100,lr=1e-5,lr_step=1e-6,lam0=1,lam=0.2,batch_size=128,threshold_d=0,tol=1e-5,display=True):
#default setting, works well for most cases
# def main(param=None,device='cuda:0',opt_mode='PGD',num_sample=100,noiselevel=0,Epoch=100,Epoch0=100,lr=4e-6,lr_step=1e-6,lam0=0.8,lam=0.1,batch_size=128,threshold_d=0):
#optuna best setting
# def main(param=None,device='cuda:0',opt_mode='PGD',num_sample=73,noiselevel=0,Epoch=231,Epoch0=348,lr=1.1e-6,lr_step=6e-6,lam0=0.8,lam=0.248,batch_size=256):
# device = 'cuda:7'
    if param is None:
        param = {}
        param['L'] = 1
        param['M'] = 1
        param['m'] = 0.5
        param['b1'] = 0.5
        param['b2'] = 0.5
        param['tau'] = 0.1
        param['omega'] = 1
        param['phi'] = 0
        param['g'] = 9.81
# The gravitational acceleration (m.s-2).
    
    cartPendulum = cartPendulum2_wrapper(param)
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
            theta = np.random.uniform(-np.pi, np.pi)
            thetadot = np.random.uniform(0,0)

            
            y0=np.array([theta,thetadot,0,0])
            x,xdot = generate_data(cartPendulum,t,y0)
            X.append(x)
            Xdot.append(xdot)
        X = np.vstack(X)
        Xdot = np.vstack(Xdot)
        #genrate sinodusal input using the omega and tau0
        Tau = np.array([np.zeros_like(t),param['tau']*np.cos(param['omega']*t)])
        Tau = torch.tensor(Tau,device=device).float()
        #duplicate the input to match the size of the data
        Tau_temp = Tau
        for i in range(num_sample-1):
            Tau_temp = torch.cat((Tau_temp, Tau), dim=1)
        Tau = Tau_temp
        Tau_org = Tau.clone()
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


    #Separating states of pendulum and cart
    pendulum_states = []
    cartpole_states = []
    for i in range(states_dim):
        if(i%2==0):
            pendulum_states.append(states[i])
        else:
            cartpole_states.append(states[i])

    #build function expression for the library in str
    pend_terms = HL.buildFunctionExpressions(1,states_dim//2,pendulum_states,use_sine=True)
    cartpole_terms = HL.buildFunctionExpressions(1,states_dim//2,cartpole_states,use_sine=False)

    #Assuming we get a prior knowledge about a single pendulum equations
    temp = pend_terms[1:] + cartpole_terms
    expr = HL.buildFunctionExpressions(3,len(temp),temp)
    print(len(expr))
    print(expr)
    exit()
    d_expr = ['x0_t**2','x1_t**2']
    if display:
        print("Expression : ", expr)
    # expr = ['cos(x0)','cos(x1)','x0_t*x1_t*cos(x0)*cos(x1)','x0_t*x1_t*sin(x0)*sin(x1)','x0_t**2','x1_t**2']


    #Creating library tensor
    Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=True)


    expr = np.array(expr)
    i0 = np.where(expr == 'x0_t**2')[0]
    i1 = np.where(expr == 'cos(x0)')[0]
    i2 = np.where(expr == 'x0_t**2*cos(x0)')[0]
    idx = np.arange(0,len(expr))
    delete_idx = [i0,i2]
    idx = np.delete(idx,delete_idx)
    known_expr = expr[i0].tolist()  
    expr = np.delete(expr,delete_idx).tolist()
    #non-penalty index from prev knowledge
    


    nonpenaltyidx = [i1]
    # nonpenaltyidx = []







    #Moving to Cuda

    Zeta_ = Zeta[:,:,i0,:].clone().detach()
    Eta_ = Eta[:,:,i0,:].clone().detach()
    Delta_ = Delta[:,i0,:].clone().detach()

    Zeta = Zeta[:,:,idx,:]
    Eta = Eta[:,:,idx,:]
    Delta = Delta[:,idx,:]
    Dissip = Dissip.to(device)
    Zeta = Zeta.to(device)
    Eta = Eta.to(device)
    Delta = Delta.to(device)

    Zeta_ = Zeta_.to(device)
    Eta_ = Eta_.to(device)
    Delta_ = Delta_.to(device)



    xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10)
    prevxi_L = xi_L.clone().detach()
    xi_d = torch.ones(len(d_expr), device=device)
    c = torch.ones(len(known_expr), device=device)

    def PGD_loop(Tau, c, coef, prevcoef,d_coef, RHS, LHS, Dissip, xdot, bs, lr, lam, momentum=True, D_CAL=False,device='cuda:0'):
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
                dhat = (d + ((i-1)/(i+2))*(d - pre_d)).clone().detach().requires_grad_(True)
                dhat = d.requires_grad_(True).clone().detach().requires_grad_(True)
            else:
                vhat = v.requires_grad_(True).clone().detach().requires_grad_(True)
                dhat = d.requires_grad_(True).clone().detach().requires_grad_(True)
    
            prev = v
            pre_d = d

            #Computing loss
            zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
            eta = Eta[:,:,:,i*bs:(i+1)*bs]
            delta = Delta[:,:,i*bs:(i+1)*bs]

            zeta_ = Zeta_[:,:,:,i*bs:(i+1)*bs]
            eta_ = Eta_[:,:,:,i*bs:(i+1)*bs]
            delta_ = Delta_[:,:,i*bs:(i+1)*bs]

            dissip = Dissip[:,:,i*bs:(i+1)*bs]
            tau = Tau[:,i*bs:(i+1)*bs]
            x_t = xdot[i*bs:(i+1)*bs,:]



            #forward
            pred = -ELforward(vhat,zeta,eta,delta,x_t,device)
            if D_CAL:
                disp = DPforward(dhat,dissip,device)
                targ = ELforward(c,zeta_,eta_,delta_,x_t,device)+disp-tau
            else:
                targ = ELforward(c,zeta_,eta_,delta_,x_t,device) + tau
            lossval = loss(pred, targ)
            
            #Backpropagation
            lossval.backward()
            with torch.no_grad():
                v = vhat - lr * vhat.grad
                v = (proxL1norm(v, lr*lam, nonpenaltyidx))
                if D_CAL:
                    d = dhat - lr * dhat.grad
                    d = (proxL1norm(d, lr*lam, nonpenaltyidx))
                #reset gradient
                vhat.grad = None
                dhat.grad = None

            
        
            
            loss_list.append(lossval.item())
        if display:
            print("Average loss : " , torch.tensor(loss_list).mean().item())
        return v, prevcoef,d, torch.tensor(loss_list).mean().item()

   

    i = 0
    temp = 1000
    RHS = [Zeta, Eta, Delta]
    LHS = [Zeta_, Eta_, Delta_]
    while(i<=Epoch0):
        if display:
            print("\n")
            print("Stage 1")
            print("Epoch "+str(i) + "/" + str(Epoch0))
            print("Learning rate : ", lr)
        xi_L, prevxi_L,xi_d, lossitem= PGD_loop(Tau, c, xi_L,prevxi_L,xi_d, RHS, LHS, Dissip, Xdot, batch_size, lr=lr,lam=lam0,momentum=True,device=device)
        temp = lossitem
        i+=1
        if display:
            print("sanity check", sanity_check(expr))
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
        print("Result stage 1: ", simplify(L))

    last_ten_loss = []
    converged = False
    ## Next round selection ##
    for stage in range(20):

        #Redefine computation after thresholding
        expr.append(known_expr[0])
        Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=False)
        expr = np.array(expr)
        i0 = np.where(expr == 'x0_t**2')[0]
        i1 = np.where(expr == 'cos(x0)')[0]
        idx = np.arange(0,len(expr))
        idx = np.delete(idx,i0)
        known_expr = expr[i0].tolist()  
        expr = np.delete(expr,i0).tolist()

 
        

        nonpenaltyidx = [i1]

        Zeta_ = Zeta[:,:,i0,:].clone().detach()
        Eta_ = Eta[:,:,i0,:].clone().detach()
        Delta_ = Delta[:,i0,:].clone().detach()

        Zeta = Zeta[:,:,idx,:]
        Eta = Eta[:,:,idx,:]
        Delta = Delta[:,idx,:]



        # nonpenaltyidx = []

        Zeta = Zeta.to(device)
        Eta = Eta.to(device)
        Delta = Delta.to(device)
        Zeta_ = Zeta_.to(device)
        Eta_ = Eta_.to(device)
        Delta_ = Delta_.to(device)

        Dissip = Dissip.to(device)


      
        i = 0
        
        # if(len(xi_L)+len(xi_d) <= 6):
        if len(xi_L) <= 3:
            lam = 0
            threshold = 1e-3
            converged = True
            Epoch = 200
        # elif(len(xi_L) <= 8):
        #     lam = 0
        else:
            threshold = 0.1
            lr += lr_step
            lam = lam
        temp = 1000
        RHS = [Zeta, Eta, Delta]
        LHS = [Zeta_, Eta_, Delta_]
        while(i<=Epoch):
            if display:
                print("\n")
                print("Stage " + str(stage+2))
                print("Epoch "+str(i) + "/" + str(Epoch))
                print("Learning rate : ", lr)
            xi_L, prevxi_L,xi_d, lossitem= PGD_loop(Tau, c, xi_L,prevxi_L,xi_d, RHS, LHS, Dissip, Xdot, batch_size, lr=lr,lam=lam,momentum=True,device=device,D_CAL=True)
            i+=1
            if display:
                print('xi_L', xi_L)
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
            if math.isnan(lossitem):
                return xi_L,100
            if(temp <= 1e-3):
                break
        
        
        ## Thresholding
        if stage < 1 or len(xi_L) > 18:
            #regularize the biggest coefficient to 20
            idx = torch.argmax(torch.abs(xi_L))
            xi_Ltemp = xi_L / xi_L[idx] * 19.6
            xi_d = xi_d / xi_L[idx] * 19.6
            Tau = Tau / xi_L[idx] * 19.6
            surv_index = ((torch.abs(xi_Ltemp) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
            expr = np.array(expr)[surv_index].tolist()

            xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
            xi_d = xi_d.clone().detach().requires_grad_(True)
            prevxi_L = xi_L.clone().detach()

            ## obtaining analytical model
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-2)
            D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
            # print("Result stage " + str(stage+2) + ":" , simplify(L))
            if display:
                print("Result stage " + str(stage+2) + ":" , L)
                print("simplified : ", simplify(L))
                print("Dissipation : ", simplify(D))
            if converged:
                break
        else:
            if display:
                print("thresholding using the simplified expression")
        ## Thresholding
            ## obtaining analytical model
            #calculate the relative threshold
            scaler = 19.6 / torch.abs(xi_L).max().item()
            xi_L = xi_L * scaler
            xi_d = xi_d * scaler
            Tau = Tau * scaler
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
            L_simplified = simplify(L)
            x0, x1,x0_t,x1_t = symbols('x0 x1 x0_t x1_t')
            coeff_dict = L_simplified.as_coefficients_dict()
            scaler = coeff_dict['cos(x0)']/20
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
            if not sanity_check(expr):
                if display:
                    print("sanity check failed")
                break
            if converged:
                total_epoch = (stage+1) * Epoch + Epoch0
                break


    ## Adding known terms
    expr = np.array(expr)
    expr = np.append(expr, known_expr)
    scaler = Tau_org[0,5]/Tau[0,5]
    xi_L = torch.cat((xi_L, c), dim=0)
    xi_L = xi_L * scaler
    xi_d = xi_d * scaler
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
    L = HL.generateExpression(xi_Lcpu,expr)
    
    L = str(simplify(L)) 
    D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
    if display:
        print("\m")
        print("Obtained Lagrangian : ", L)
        print("Obtained Dissipation : ", simplify(D))
    #caluclate the relative error of the obtained coefficients
    #the real Lagrangian model is m1*l1**2*x0_t**2/2 + m2*(l1**2*x0_t**2/2 + l2**2*x1_t**2/2 + l1*l2*x0_t*x1_t*cos(x0)*cos(x1)+l1*l2*x0_t*x1_t*sin(x0)*sin(x1)) + (m1+m2)*g*l1*cos(x0) + m2*g*l2*cos(x1)

    l, M, m, g = param['L'], param['M'], param['m'] , param['g']

    # Define the symbols
    x0, x1, x0_t, x1_t = symbols('x0 x1 x0_t x1_t')

    # Define the real Lagrangian model
    L_real = 0.5*(M+m)*x1_t**2+m*l*x0_t*x1_t*cos(x0)+0.5*m*l**2*x0_t**2+m*g*l*cos(x0)



    # Get the real coefficients
    real_coeff_dict = L_real.as_coefficients_dict()
    real_coeff_dict = {str(key): val for key, val in real_coeff_dict.items()}

    # Create a dictionary of estimated coefficients
    estimated_coeff_dict = filter_dict
    estimated_coeff_dict['x0_t**2'] = float(1.0)
    #change the keys of the estimated_coeff_dict to string
    estimated_coeff_dict = {str(key): val for key, val in estimated_coeff_dict.items()}
    
    #scale the x0_t**2 and use that scaler to scale the other coefficients
    scale = real_coeff_dict['x0_t**2']/estimated_coeff_dict['x0_t**2']

    for key in estimated_coeff_dict.keys():
        estimated_coeff_dict[key] = estimated_coeff_dict[key]*scale

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
            if display:
                print(f"The term {cand} is not in the real coefficients")
            sum_relative_errors += 1

    # Print the relative errors
    if display:
        print("The relative errors are:", sum_relative_errors)
    





    if(save==True):
        #Saving Equation in string
        text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
        text_file.write(L)
        text_file.close()
    return estimated_coeff_dict, sum_relative_errors, total_epoch
if __name__ == "__main__":
    main()



