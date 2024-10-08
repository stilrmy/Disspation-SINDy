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



import time

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])

def def main(param=None,device='cuda:0',num_sample=100,noiselevel=0,Epoch=231,Epoch0=348,lr=1.1e-6,lr_step=6e-6,lam0=0.8,lam=0.248,batch_size=256):

    device = 'cuda:2'
    param = {}
    param['L1'] = 1
    param['L2'] = 1
    param['m1'] = 1
    param['m2'] = 1
    param['b1'] = 0
    param['b2'] = 0
    param['tau0'] = 0.5
    param['omega1'] = 0.5*np.pi
    param['omega2'] = np.random.uniform(0,np.pi)
    param['phi'] = 0
    # The gravitational acceleration (m.s-2).
    g = 9.81
    opt_mode = "adaPGM"



    def doublePendulum2_wrapper(params):
        def doublePendulum2(t, y):
            l1, l2, m1, m2, b1, b2, tau0, omega1, omega2, phi = params['L1'], params['L2'], params['m1'], params['m2'], params['b1'], params['b2'], params['tau0'], params['omega1'], params['omega1'], params['phi']

            q1,q2,q1_t,q2_t = y

            q1_2t = (m2*l1*q1_t**2*np.sin(2*(q1-q2))+2*m2*l2*q2_t**2*np.sin(q1-q2)+2*g*m2*np.cos(q2)*np.sin(q1-q2)+2*g*m1*np.sin(q1)+(2*b1*q1_t-2*tau0*np.cos(omega1*t)-2*np.cos(q1-q2)*(b2*q2_t-tau0*np.cos(omega2*t))))/(-2*l1*(m1+m2*(np.sin(q1-q2)**2)))

            q2_2t = (m2*l2*q2_t**2*np.sin(2*(q1-q2))+2*(m1+m2)*l1*q1_t**2*np.sin(q1-q2)+2*(m1+m2)*g*np.cos(q1)*np.sin(q1-q2)+2*(b1*q1_t-tau0*np.cos(omega1*t))*np.cos(q1-q2)-(2*(m1+m2)/m2)*(b2*q2_t-tau0*np.cos(omega2*t)))/(2*l2*(m1+m2*(np.sin(q1-q2)**2)))


            return q1_t, q2_t, q1_2t, q2_2t
        return doublePendulum2

    doublePendulum = doublePendulum2_wrapper(param)


    #Saving Directory
    rootdir = "../Double Pendulum/Data/"

    num_sample = 10
    create_data = True
    training = True
    save = False
    noiselevel = 0


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
        #genrate sinodusal input using the omega and tau0
        Tau = np.array([param['tau0']*np.cos(param['omega1']*t + param['phi']),param['tau0']*np.cos(param['omega1']*t + param['phi'])])
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
    # d_expr = ['x0_t**2','x1_t**2']
    d_expr = []
    # expr = ['cos(x0)','cos(x1)','x0_t*x1_t*cos(x0)*cos(x1)','x0_t*x1_t*sin(x0)*sin(x1)','x0_t**2','x1_t**2']


    #Creating library tensor
    Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=False)


    expr = np.array(expr)


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




    xi_L = torch.ones(len(expr), device=device)
    prevxi_L = xi_L.clone().detach()
    xi_d = torch.ones(len(d_expr), device=device).data.uniform_(-20,20)



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


    def prox(x, gamma=1, lambda_=1, nonpenaltyidx=None):
        # Compute gamma * lambda
        gl = gamma * lambda_
        
        # Apply the proximal operator conditions in a vectorized manner
        y = torch.where(x <= -gl, x + gl, torch.where(x >= gl, torch.zeros_like(x), -x))
        
        # Apply nonpenalty indices: set elements of y at nonpenaltyidx to their original values in x
        if nonpenaltyidx is not None:
            y[nonpenaltyidx] = x[nonpenaltyidx]
        
        
        return y

    #ADAM optimizer
    def SR_loop(Tau, coef, prevcoef, d_coef, RHS, Dissip, xdot, bs, lr, lam,beta1=0.9,beta2=0.999,eps=1e-8):
        predcoef = coef.clone().detach().to(device).requires_grad_(True)
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
            tau = Tau[:,i*bs:(i+1)*bs]
            x_t = xdot[i*bs:(i+1)*bs,:]
            vdhat_ = torch.tensor([0.025,0.025]).to(device).float()
            #forward
            vdhat_ = torch.zeros_like(vdhat)
            disp = DPforward(vdhat_,dissip,device)
            pred = -tauforward(vhat,zeta,eta,delta,x_t,device)
            targ = tau 
            lossval = loss(pred, targ)
            #lossval = torch.mean(lossval[0,:]**2)+torch.mean(lossval[1,:]**2)
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
        print("Average loss : " , torch.tensor(loss_list).mean().item())
        return v, prev, vdhat_, torch.tensor(loss_list).mean().item()

    #PGD optimizer
    def PGD_loop(Tau, coef, prevcoef, d_coef, RHS, Dissip, xdot, bs, lr, lam, momentum=True, D_CAL=False, device='cuda:0', it=100):
        loss_list = []
        tl = xdot.shape[0]
        Zeta, Eta, Delta = RHS
        if not torch.is_tensor(xdot):
            xdot = torch.from_numpy(xdot).to(device).float()
        
        v = coef.clone().detach().requires_grad_(True)
        d = d_coef.clone().detach().requires_grad_(True)
        # Parameters to optimize
        params_to_optimize = [v]
        if D_CAL:
            
            params_to_optimize.append(d)
        
        # Initialize optimizer with the dynamic set of parameters
        optimizer = torch.optim.SGD(params_to_optimize, lr=lr, momentum=0.9 if momentum else 0.0)
    
        for epoch in range(it):

            for i in range(tl // bs):
                optimizer.zero_grad()  # Reset gradients for this iteration

                # Slicing for current batch
                zeta = Zeta[:, :, :, i*bs:(i+1)*bs]
                eta = Eta[:, :, :, i*bs:(i+1)*bs]
                delta = Delta[:, :, i*bs:(i+1)*bs]
                dissip = Dissip[:, :, i*bs:(i+1)*bs]
                tau = Tau[:, i*bs:(i+1)*bs]
                x_t = xdot[i*bs:(i+1)*bs, :]

                # Forward pass (example function calls, implement according to your needs)
                pred, weight = ELDPforward(v, d, zeta, eta, delta, dissip, x_t, device, D_CAL)
                targ = tau
                lossval = loss(pred, targ)

                # Backward pass and optimizer step
                lossval.backward()
                optimizer.step()

                loss_list.append(lossval.item())
            if epoch%10 == 0:
                print("Epoch "+str(epoch) + "/" + str(it))
                print("Learning rate : ", lr)
                print("Average loss:", torch.tensor(loss_list).mean().item())
        return v, d , torch.tensor(loss_list).mean().item()




    # def stepsize(lr,pre_lr, x_grad, pre_x_grad, x, pre_x):
    #     # Calculate the Lipschitz constant 'l_k'
    #     l_k_numerator = (x_grad - pre_x_grad) * (x - pre_x)
    #     l_k = l_k_numerator.sum() / torch.norm(x - pre_x, p=2)**2

    #     # Calculate the curvature constant 'c_k'
    #     c_k_numerator = torch.norm(x_grad - pre_x_grad, p=2)**2
    #     c_k = c_k_numerator / l_k_numerator.sum()

        
    #     # Compute gamma_k, assume gamma_0 is given as 'lr' (learning rate)
    #     gamma_k = lr
        
    #     # Calculate gamma_k+1 using the formula
    #     term1 = torch.sqrt(torch.tensor(1.0) + gamma_k / pre_lr)
    #     term2 = 1.0 / (2.0 * torch.sqrt(gamma_k * l_k * (gamma_k * c_k - 1.0)) + 1e-8)  # Add epsilon for numerical stability
    #     minimum = torch.where(torch.isnan(term2), term1, torch.min(term1, term2))
    #     print('minimum', minimum)
    #     gamma_k_plus_1 = gamma_k * minimum
    #     return gamma_k_plus_1
    def stepsize(pre_lr,pre_pre_lr, x_grad, pre_x_grad, x, pre_x):
        theta = torch.tensor(1.2,device=device)
        t = torch.tensor(1.0,device=device)
        norm_A = torch.tensor(0,device=device)
        delta = torch.tensor(0,device=device)
        if(torch.is_tensor(pre_lr)==False):
            pre_lr = torch.tensor(pre_lr,device=device)
        if(torch.is_tensor(pre_pre_lr)==False):
            pre_pre_lr = torch.tensor(pre_pre_lr,device=device)
        xi = t**2 * pre_lr * norm_A**2
        C = torch.norm(x_grad - pre_x_grad, p=2)**2/torch.dot(x_grad - pre_x_grad, x - pre_x)
        if torch.isnan(C):
            C = 0
        L = torch.dot(x_grad - pre_x_grad, x - pre_x)/torch.norm(x - pre_x, p=2)**2
        if torch.isnan(L):
            L = 0
        D = pre_lr * L * (pre_lr * C - 1)
        if torch.isnan(D):
            D = 0
        print(torch.dot(x_grad - pre_x_grad, x - pre_x))
        lr = torch.min(pre_lr * torch.sqrt(1+pre_lr/pre_pre_lr), torch.min(1/(2 * theta * t * norm_A + 1e-8), (pre_lr * torch.sqrt(1 - 4 * xi * (1 + delta)**2)/torch.sqrt(2*(1 + delta)*(D * torch.sqrt(D**2 + xi * (1 - 4 * xi * (1 + delta)**2)))))))
        return lr


    def adaPGM(Tau, coef,d_coef, RHS, Dissip, xdot, it, lam,D_CAL=False,device='cuda:0'):
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

            #run a initial run to get the gradient
            pred, weight,candidate = adaELDPforward(v,d,Zeta,Eta,Delta,Dissip,xdot,device,D_CAL)
            # lr0 = 1/(torch.norm(candidate,2)**2)
            lr0 = 1e-11
            targ = Tau
            lossval = loss(pred, targ)
            lossval.backward()
            with torch.no_grad():
                pre_lr = lr0
                pre_pre_lr = lr0
                lr = lr0
                pre_v_grad = weight.grad.clone().detach()
                pre_v = weight.clone().detach()
                weight = weight - lr * weight.grad
                weight = proxL1norm(weight,lr*lam,nonpenaltyidx)
                if D_CAL:
                    v = weight[:len(expr)].clone().detach().requires_grad_(True)
                    d = weight[len(expr):].clone().detach().requires_grad_(True)
                else:
                    v = weight.clone().detach().requires_grad_(True)
                weight.grad = None
            for i in range(it):            

                #forward
                pred, weight = ELDPforward(v,d,Zeta,Eta,Delta,Dissip,xdot,device,D_CAL)
                targ = Tau
                lossval = loss(pred, targ) + lam * torch.norm(v,1)
                
                #Backpropagation
                lossval.backward()
                with torch.no_grad():
                    
                    # lr = stepsize(pre_lr,pre_pre_lr,weight.grad, pre_v_grad,v,pre_v)
                    lr = 1e-7
                    pre_pre_lr = pre_lr
                    pre_lr = lr
                    pre_v_grad = weight.grad.clone().detach()
                    if D_CAL:
                        pre_v = weight[:len(expr)].clone().detach()
                    else:
                        pre_v = weight.clone().detach()
                    weight = weight - lr * pre_v_grad
                    weight = proxL1norm(weight,lr*lam,nonpenaltyidx)
                    weight.grad = None
                    if D_CAL:
                        v = weight[:len(expr)].clone().detach().requires_grad_(True)
                        d = weight[len(expr):].clone().detach().requires_grad_(True)
                    else:
                        v = weight.clone().detach().requires_grad_(True)
                    
            
                
                loss_list.append(lossval.item())
                if i%10 == 0:
                    print("Iteration : ", i, " Loss : ", lossval.item())
                    print("Learning rate : ", lr)
            return v,d, torch.tensor(loss_list).mean().item()




    #adaPGM optimizer

    #training loop that return different optimizer based on the global opt_mode variable

    Epoch = 200
    i = 0

    lam = 1
    lr = 5e-6
    temp = 1000
    it = 10000
    RHS = [Zeta, Eta, Delta]

        # xi_L, xi_d, lossitem= adaPGM(Tau,xi_L,xi_d,RHS,Dissip,Xdot,it,lam=lam,D_CAL=False,device=device)
    print("\n")
    print("Stage 1")    
    xi_L,xi_d,lossitem = PGD_loop(Tau,xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,bs=100,lr=lr,lam=lam,momentum=True,D_CAL=False,device=device,it=Epoch)
    temp = lossitem
    i+=1


    ## Thresholding
    threshold = 1e-1
    surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
    xi_d = xi_d.clone().detach().requires_grad_(True)
    prevxi_L = xi_L.clone().detach()

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-3)
    print("Result stage 1: ", simplify(L))



        
        
    ## Thresholding
    for stage in range(9):

        #Redefine computation after thresholding
        
        Zeta, Eta, Delta, Dissip = LagrangianLibraryTensor(X,Xdot,expr,d_expr,states,states_dot, scaling=False)


        expr = np.array(expr)
        i1 = np.where(expr == 'x0_t**2')[0]
        i4 = np.where(expr == 'x1_t**2')[0][0]
        i5 = np.where(expr == 'cos(x0)')[0][0]
        i6 = np.where(expr == 'cos(x1)')[0][0]
        idx = np.arange(0,len(expr))
        

        nonpenaltyidx = [i4,i5,i6]



        Zeta = Zeta[:,:,idx,:]
        Eta = Eta[:,:,idx,:]
        Delta = Delta[:,idx,:]

        # nonpenaltyidx = []

        Zeta = Zeta.to(device)
        Eta = Eta.to(device)
        Delta = Delta.to(device)
        Dissip = Dissip.to(device)


        
        i = 0
        
        if(len(xi_L) <= 8):
            lam = 0
            threshold = 1e-3
        # elif(len(xi_L) <= 8):
        #     lam = 0
        else:
            threshold = 0.01
            lr_step = 1e-6
            lr += lr_step
            lam = 0.3
        temp = 1000
        RHS = [Zeta, Eta, Delta]
        batch_size = 100
        print("\n")
        print("Stage " + str(stage+2) + " :") 
        xi_L,xi_d,lossitem = PGD_loop(Tau,xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,bs=100,lr=lr,lam=lam,momentum=True,D_CAL=False,device=device,it=Epoch)
        temp = lossitem
        
        
        ## Thresholding
        if stage < 100:
            
            #regularize the biggest coefficient to 20
            idx = torch.argmax(torch.abs(xi_L))
            xi_Ltemp = xi_L / xi_L[idx] * 20
            surv_index = ((torch.abs(xi_Ltemp) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
            expr = np.array(expr)[surv_index].tolist()

            xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
            xi_d = xi_d.clone().detach().requires_grad_(True)
            prevxi_L = xi_L.clone().detach()
            print(xi_L)
            ## obtaining analytical model
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            # D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
            # print("Result stage " + str(stage+2) + ":" , simplify(L))
            print("Result stage " + str(stage+2) + ":" , L)
            print("simplified : ", simplify(L))
            # print("Dissipation : ", simplify(D))
        else:
            print("thresholding using the simplified expression")
        ## Thresholding
            ## obtaining analytical model
            #calculate the relative threshold
            scaler = 20 / torch.abs(xi_L).max().item()
            xi_L = xi_L * scaler
            xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            # D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
            L_simplified = simplify(L)
            x0, x1,x0_t,x1_t = symbols('x0 x1 x0_t x1_t')
            coeff_dict = L_simplified.as_coefficients_dict()
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

            L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-1)
            print("Result stage " + str(stage+2) + ":" , L)
            # print("Dissipation : ", simplify(D))



    ## Adding known terms
    L = str(simplify(L)) 
    D = HL.generateExpression(xi_d.detach().cpu().numpy(),d_expr)
    print("\m")
    print("Obtained Lagrangian : ", L)
    print("Obtained Dissipation : ", simplify(D))




    if(save==True):
        #Saving Equation in string
        text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
        text_file.write(L)
        text_file.close()





