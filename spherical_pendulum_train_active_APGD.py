import numpy as np
import sys 

from tqdm import tqdm
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
import os


import time

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])

def sphericalPendulum2_wrapper(params):
    def sphericalPendulum2(t, y):
        g,L,m,b_theta,b_phi,F_theta,F_phi = params['g'],params['L'],params['m'],params['b_theta'],params['b_phi'],params['F_theta'],params['F_phi']

        x0,x1,x0_t,x1_t = y

        F_Theta = F_theta * np.cos(t)
        F_Phi = F_phi * np.sin(t)

        x0_2t = sin(x0) * cos(x0) * x1_t**2 - (g / L) * sin(x0) - (b_theta / (m * L**2)) * x0_t + (F_Theta / (m * L**2))
        


        x1_2t = -2 * (cos(x0) / sin(x0)) * x0_t * x1_t - (b_phi / (m * L**2 * sin(x0)**2)) * x1_t + (F_Phi / (m * L**2 * sin(x0)**2))
        


        return x0_t, x1_t, x0_2t, x1_2t
    return sphericalPendulum2

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




def main(param=None,device='cuda:1',opt_mode='PGD',num_sample=2,noiselevel=0,Epoch=100,Epoch0=100,lr=4e-6,lr_step=1e-6,lam0=0.8,lam=0.1,batch_size=128,threshold_d=1e-6,tol=1e-5,display=True):

#default setting, works well for most cases
# def main(param=None,device='cuda:0',opt_mode='PGD',num_sample=100,noiselevel=0,Epoch=100,Epoch0=100,lr=4e-6,lr_step=1e-6,lam0=0.8,lam=0.1,batch_size=128,threshold_d=0):
#optuna best setting
# def main(param=None,device='cuda:0',opt_mode='PGD',num_sample=73,noiselevel=0,Epoch=231,Epoch0=348,lr=1.1e-6,lr_step=6e-6,lam0=0.8,lam=0.248,batch_size=256):
# device = 'cuda:7'
    if param is None:
        param = {}
        param['g'] = 9.81
        param['L'] = 1
        param['m'] = 1
        param['b_theta'] = 0.01
        param['b_phi'] = 0.01
        param['F_theta'] = 0.1
        param['F_phi'] = 0.1
# The gravitational acceleration (m.s-2).
    
    sphericalPendulum = sphericalPendulum2_wrapper(param)
    #Saving Directory
    rootdir = ".../spherical_pendulum/"
    create_data = True
    training = True
    save = False

    if(create_data):
        if display:
            print("Creating Data")
        X, Xdot = [], []


        for i in tqdm(range(num_sample)):
            t = np.arange(0,5,0.01)
            theta = np.random.uniform(np.pi/3, np.pi/2)
            
            
            y0=np.array([theta,0,0,np.pi])
            x,xdot = generate_data(sphericalPendulum,t,y0)
            X.append(x)
            Xdot.append(xdot)
        X = np.vstack(X)
        Xdot = np.vstack(Xdot)
        #genrate sinodusal input using the omega and tau0
        Tau = np.array([param['F_theta']*np.cos(t),param['F_phi']*np.sin(t)])
        Tau = torch.tensor(Tau,device=device).float()
        #duplicate the input to match the size of the data
        Tau_temp = Tau
        for i in range(num_sample-1):
            Tau_temp = torch.cat((Tau_temp, Tau), dim=1)
        Tau = Tau_temp
        Tau_org = Tau.clone()
        if(save==True):
            #create the folder if not exist
            if not os.path.exists(rootdir):
                os.makedirs(rootdir)
            np.save(rootdir + "spherical_X.npy", X)
            np.save(rootdir + "spherical_Xdot.npy",Xdot)
            Tau = Tau.cpu().numpy()
            np.save(rootdir + "spherical_Tau.npy",Tau)
            Tau = torch.tensor(Tau,device=device).float()
    else:
        X = np.load(rootdir + "spherical_X.npy")
        Xdot = np.load(rootdir + "spherical_Xdot.npy")
        Tau = np.load(rootdir + "spherical_Tau.npy")
        Tau = torch.tensor(Tau,device=device).float()
        Tau_org = Tau.clone()

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
    polynom = exprdummy[1:4]
    trig = [exprdummy[4], exprdummy[6]]
    polynom = HL.buildFunctionExpressions(2,len(polynom),polynom)
    trig = HL.buildFunctionExpressions(2, len(trig),trig)
    product = []
    for p in polynom:
        for t in trig:
            product.append(p + '*' + t)
    expr = polynom + trig + product
    print(len(expr))
    print(expr)
    exit()
    # d_expr = ['x0_t**2','x1_t**2','x0_t','x1_t']
    d_expr = ['x0_t**2','x1_t**2']
    # expr = ['cos(x0)','cos(x1)','x0_t*x1_t*cos(x0)*cos(x1)','x0_t*x1_t*sin(x0)*sin(x1)','x0_t**2','x1_t**2']


    #Creating library tensor
    Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=True)


    expr = np.array(expr)
    i1 = np.where(expr == 'x0_t**2')[0]
    i2 = np.where(expr == 'x0_t**2*cos(x0)**2')[0]

    idx = np.arange(0,len(expr))
    idx = np.delete(idx,[i2])
    expr = np.delete(expr,[i2])

    #non-penalty index from prev knowledge
    i3 = np.where(expr == 'cos(x0)')[0][0]
    nonpenaltyidx=[i3]

    expr = expr.tolist()



    Zeta = Zeta[:,:,idx,:]
    Eta = Eta[:,:,idx,:]
    Delta = Delta[:,idx,:]



    #Moving to Cuda


    Zeta = Zeta.to(device)
    Eta = Eta.to(device)
    Delta = Delta.to(device)
    Dissip = Dissip.to(device)




    xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10)
    prevxi_L = xi_L.clone().detach()
    xi_d = torch.ones(len(d_expr), device=device).data.uniform_(-5,5)


    def PGD_loop(Tau, coef, prevcoef,d_coef, RHS, Dissip, xdot, bs, lr, lam, momentum=True, D_CAL=False,device='cuda:0'):
        loss_list = []
        tl = xdot.shape[0]
        n = xdot.shape[1]
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

            #Computing loss
            zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
            eta = Eta[:,:,:,i*bs:(i+1)*bs]
            delta = Delta[:,:,i*bs:(i+1)*bs]

            dissip = Dissip[:,:,i*bs:(i+1)*bs]
            tau = Tau[:,i*bs:(i+1)*bs]
            x_t = xdot[i*bs:(i+1)*bs,:]



            #forward
            disp = DPforward(dhat,dissip,device)
            pred, weight = ELDPforward(vhat,dhat,zeta,eta,delta,dissip,x_t,device,D_CAL)
            targ = tau 
            lossval = loss(pred, targ)
            
            #Backpropagation
            lossval.backward()
            with torch.no_grad():
                weight = weight - lr * weight.grad
                weight[:len(expr)] = proxL1norm(weight[:len(expr)], lr*lam, nonpenaltyidx)
                weight.grad = None
                if D_CAL:
                    v = weight[:len(expr)]
                    d = weight[len(expr):]
                else:
                    v = weight
            
        
            
            loss_list.append(lossval.item())
        if display:
            print("Average loss : " , torch.tensor(loss_list).mean().item())
        return v, prevcoef,d, torch.tensor(loss_list).mean().item()



    i = 0
    temp = 1000
    RHS = [Zeta, Eta, Delta]
    loss_log = []
    while(i<=Epoch0):
        if display:
            print("\n")
            print("Stage 1")
            print("Epoch "+str(i) + "/" + str(Epoch0))
            print("Learning rate : ", lr)
        xi_L, prevxi_L, xi_d, lossitem= PGD_loop(Tau, xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,batch_size,lr=lr,lam=lam0,device=device,D_CAL=False)
        temp = lossitem
        loss_log.append(lossitem)
        i+=1
        if math.isnan(temp):
            return 100,0


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
    quiting = 0
    ## Next round selection ##
    for stage in range(40):

        #Redefine computation after thresholding
        
        Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=False)


        expr = np.array(expr)
        i4 = np.where(expr == 'cos(x0)')[0][0]

        nonpenaltyidx = [i4]
        expr = expr.tolist()









        Zeta = Zeta.to(device)
        Eta = Eta.to(device)
        Delta = Delta.to(device)
        Dissip = Dissip.to(device)


      
        i = 0
        
        if(len(xi_L)+len(xi_d) <= 5):
            lam = 0
            threshold = 1e-3
            converged = True
            quiting = 2
            Epoch = 100
        # elif(len(xi_L) <= 8):
        #     lam = 0
        else:
            threshold = 0.01
            lr += lr_step
            lam = lam
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
            xi_L, prevxi_L, xi_d, lossitem= PGD_loop(Tau, xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,batch_size,lr=lr,lam=lam,D_CAL=True,device=device)
            loss_log.append(lossitem)
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
                return 100,0
            if(temp <= 1e-3):
                break
        
        
        ## Thresholding
        if stage < 1:
            
            #regularize the biggest coefficient to 20
            idx = torch.argmax(torch.abs(xi_L))
            cof = param['m'] * param['L']*param['g']
            xi_Ltemp = xi_L / xi_L[idx] * cof
            xi_d = xi_d / xi_L[idx] * cof
            Tau = Tau / xi_L[idx] * cof
            surv_index = ((torch.abs(xi_Ltemp) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
            expr = np.array(expr)[surv_index].tolist()

            xi_L =xi_Ltemp[surv_index].clone().detach().requires_grad_(True)
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

            scaler = cof / torch.abs(xi_L).max().item()
            xi_L = xi_L * scaler
            xi_d = xi_d * scaler
            Tau = Tau * scaler
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
            L_simplified = simplify(L)
            x0, x1,x0_t,x1_t = symbols('x0 x1 x0_t x1_t')
            coeff_dict = L_simplified.as_coefficients_dict()

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
                total_epoch = (stage+1) * Epoch + Epoch0
            if converged:
                #if the training is converged, run a extra round with strict threshold to remove uneccessary terms, then break the loop
                if quiting == 2:
                    total_epoch = stage * Epoch + Epoch0 + 1000
                    break
                else:
                    quiting += 1



    
    ## Adding known terms
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

    m,g,L = param['m'],param['g'],param['L']
    # Define the symbols
    x0, x1, x0_t, x1_t = symbols('x0 x1 x0_t x1_t')

    # Define the real Lagrangian model
    L_real = 0.5*m*L**2*x0_t**2 + 0.5*m*L**2*x1_t**2*sin(x0)**2 + m*g*L*cos(x0)


    # Simplify the real Lagrangian model if x0_t*x1_t*cos(x0 - x1) appears in the estimated candidates


    # Get the real coefficients
    real_coeff_dict = L_real.as_coefficients_dict()
    # Create a dictionary of estimated coefficients
    estimated_coeff_dict = filter_dict


    #scale the x0_t**2 and use that scaler to scale the other coefficients
    scale = 0.5*m*L**2/estimated_coeff_dict[x0_t**2]

    for key in estimated_coeff_dict.keys():
        estimated_coeff_dict[key] = estimated_coeff_dict[key]*scale

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

    if display:
        print("The relative errors are:", sum_relative_errors)
    






    return  sum_relative_errors,total_epoch
if __name__ == "__main__":
    main()



