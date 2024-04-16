import numpy as np
import sys 


from sympy import symbols, simplify, derive_by_array, cos, sin, sympify
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
from optimizer import Adsgd
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


def main(param=None,device='cuda:4',num_sample=73,noiselevel=0,Epoch=231,Epoch0=348,lr=1.1e-6,lr_step=6e-6,lam0=0.8,lam=0.248,batch_size=256):
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

    
    create_data = True
    training = True
    save = False



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




    xi_L = torch.ones(len(expr), device=device).data.uniform_(-20,20)
    prevxi_L = xi_L.clone().detach()
    xi_d = torch.ones(len(d_expr), device=device)



    # Function to clone a parameter including its data and gradient
    def clone_param_with_grad(param):
        # Clone parameter data
        cloned_param = param.clone().detach().requires_grad_(True)
        # Check if the original parameter has gradients and clone them as well
        if param.grad is not None:
            cloned_param.grad = param.grad.clone()
        return cloned_param






    #PGD optimizer
    def train_loop(Tau, coef, prevcoef, d_coef, RHS, Dissip, xdot, bs, lr, amplifier=0.02,damping=1,weight_decay=0,eps=1e-8, lam=0.1, D_CAL=False, device='cuda:0', it=100):
        loss_list = []
        tl = xdot.shape[0]
        Zeta, Eta, Delta = RHS
        if not torch.is_tensor(xdot):
            xdot = torch.from_numpy(xdot).to(device).float()
        
        v = coef.clone().detach().requires_grad_(True)
        d = d_coef.clone().detach().requires_grad_(True)
        clone_v = clone_param_with_grad(v)
        clone_d = clone_param_with_grad(d)
        # Parameters to optimize
        params_to_optimize = [v]
        if D_CAL:
            params_to_optimize.append(d)

        
        
        # Initialize optimizer with the dynamic set of parameters
        optimizer = Adsgd(params_to_optimize,lr=lr,amplifier=amplifier, damping=damping, weight_decay=weight_decay,eps=eps)


        pred, weight = ELDPforward(clone_v, clone_d, Zeta, Eta, Delta, Dissip, xdot, device, D_CAL)
        targ = Tau
        lossval = torch.mean((pred - targ)**2) + lam * torch.norm(clone_v, 1)
        lossval.backward()
        params_to_optimize_prev = [clone_v]
        if D_CAL:
            params_to_optimize_prev.append(clone_d)
        prev_optimizer = Adsgd(params_to_optimize_prev, lr=lr, amplifier=amplifier, damping=damping, weight_decay=weight_decay, eps=eps)
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
                lossval = torch.mean((pred - targ)**2) + lam * torch.norm(v, 1)

                # Backward pass and optimizer step
                lossval.backward()
                optimizer.compute_dif_norms(prev_optimizer)
                #change the parameters inside the prev_optimizer to the current v and d
                clone_v = clone_param_with_grad(v)
                temp_params = [clone_v]
                if D_CAL:
                    clone_d = clone_param_with_grad(d)
                    temp_params = [clone_v, clone_d]
                prev_optimizer.param_groups[0]['params'] = temp_params
                optimizer.step()

                loss_list.append(lossval.item())
            if epoch%10 == 0:
                print("Epoch "+str(epoch) + "/" + str(it))
                print("Learning rate : ", lr)
                print("Average loss:", torch.tensor(loss_list).mean().item())
        return v, d , torch.tensor(loss_list).mean().item()









    #training loop that return different optimizer based on the global opt_mode variable

    
    i = 0
    temp = 1000
    RHS = [Zeta, Eta, Delta]

        # xi_L, xi_d, lossitem= adaPGM(Tau,xi_L,xi_d,RHS,Dissip,Xdot,it,lam=lam,D_CAL=False,device=device)
    print("\n")
    print("Stage 1")    
    xi_L,xi_d,lossitem = train_loop(Tau,xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,batch_size,lr,amplifier=0.02,damping=1,weight_decay=0,eps=1e-8, lam=lam0, D_CAL=False, device=device, it=Epoch0)
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
    for stage in range(6):

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
        lr += lr_step
        if(len(xi_L) <= 8):
            lam = 0
            threshold = 1e-3
        # elif(len(xi_L) <= 8):
        #     lam = 0
        else:
            threshold = 0.01
            lam = 0.5
        temp = 1000
        RHS = [Zeta, Eta, Delta]
        print("\n")
        print("Stage " + str(stage+2) + " :") 
        xi_L,xi_d,lossitem = train_loop(Tau,xi_L,prevxi_L,xi_d,RHS,Dissip,Xdot,batch_size,lr,amplifier=0.02,damping=1,weight_decay=0,eps=1e-8, lam=lam, D_CAL=True, device=device, it=Epoch)
        temp = lossitem
        
        
        ## Thresholding
        if stage < 4:
            
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


    if(save==True):
        #Saving Equation in string
        text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
        text_file.write(L)
        text_file.close()
    return estimated_coeff_dict, sum_relative_errors
if __name__ == "__main__":
    main()





