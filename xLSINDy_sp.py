'''written by Adam Purnomo'''

import HLsearch as HL
import numpy as np

from sympy import symbols, simplify, derive_by_array

import sympy
import torch
import torch.jit as jit
import sys
sys.path.append(r'../../HLsearch/')


def LagrangianLibraryTensor(x, xdot, expr, d_expr, states, states_dot,device, scaling=False, scales=None):
    """
    A function dedicated to build time-series tensor for the lagrangian equation.
    The lagrangian equation is described as follow
    L = sum(c_k*phi_k)
    q_tt = (D^2L_qdot2)^-1*(tau + DL_q - D^2L_qdotq)

    #Params:
    x                       : values of state variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    xdot                    : values of states_dot variables in torch tensor. In [x_dot,x_doubledot] format. Each row presents states at one time
    expr                    : list of basis function (str) (d,)
    states                  : list states variable description (str) (n,)
    states_dot              : time derivative state_variable (str) (n,)

    #Return:
    Zeta                    : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta                     : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta                   : time-series of derivative of basis functions w.r.t q 
    """
    from torch import cos, sin

    x = torch.from_numpy(x)
    xdot = torch.from_numpy(xdot)
    n = len(states)#number of states
    q = sympy.Array(np.array(sympy.Matrix(states[:n//2])).squeeze().tolist())#states(x0,x1,for double pendulum)
    qdot = sympy.Array(np.array(sympy.Matrix(states[n//2:])).squeeze().tolist())#states_dot(x0_dot,x1_dot,for double pendulum)
    phi = sympy.Array(np.array(sympy.Matrix(expr)).squeeze().tolist())
    d = sympy.Array(np.array(sympy.Matrix(d_expr)).squeeze().tolist())
    phi_q = derive_by_array(phi, q)
    # print(phi_q.shape)
    # print(q)
    # print(qdot)
    


    phi_qdot = derive_by_array(phi, qdot)
    phi_qdot2 = derive_by_array(phi_qdot, qdot)
    phi_qdotq = derive_by_array(phi_qdot, q)
    d_qdot = derive_by_array(d, qdot)
    # print("dissip_qdot",d_qdot)
    # print("phi_q",phi_q)
    # print("phi_qdot", phi_qdot)
    # print("phi_qdot2",phi_qdot2)
    # print("phi_qdotq",phi_qdotq)
    # print("dissip_qdot_shape",d_qdot.shape)

    l = x.shape[0]
    if len(np.array(phi_qdot2).shape) == 1:
        j = int(np.array(phi_qdot2).shape[0])
        Delta = torch.ones(j, l,device=device)
        Zeta = torch.ones(j, l,device=device)
        Eta = torch.ones(j, l,device=device)
    else:
        i, j, k = np.array(phi_qdot2).shape
        p,q = np.array(d_qdot).shape
        Delta = torch.ones(j, k, l,device=device)
        Zeta = torch.ones(i, j, k, l,device=device)
        Eta = torch.ones(i, j, k, l,device=device)
        Dissip = torch.ones(p,q, l,device=device)
    for idx in range(len(states)):
        locals()[states[idx]] = x[:, idx]
        "adding new variables x0/x1.... at time series"
        'combine with the eval() to compute the value'
        locals()[states_dot[idx]] = xdot[:, idx]
    if len(np.array(phi_q).shape) == 1:
            for n in range(j):
                delta = eval(str(phi_q[n]))
                zeta = eval(str(phi_qdot2[n]))
                eta = eval(str(phi_qdotq[n]))
                'time series of the value of phi_q'

            if(isinstance(delta, int)):
                Delta[n, :] = delta*Delta[n, :]
            else:
                # Feature Scaling
                if(scaling == True):
                    scales = torch.max(delta) - torch.min(delta)
                    delta = delta/scales
                Delta[n, :] = delta

            if (isinstance(zeta, int)):
                Zeta[n, :] = zeta * Zeta[n, :]
            else:
                # Feature Scaling
                if (scaling == True):
                    scales = torch.max(zeta) - torch.min(zeta)
                    zeta = zeta / scales
                Zeta[n, :] = zeta

            if (isinstance(eta, int)):
                Eta[n, :] = eta * Eta[n, :]
            else:
                # Feature Scaling
                if (scaling == True):
                    scales = torch.max(eta) - torch.min(eta)
                    eta = eta / scales
                Eta[n, :] = eta

    else:
        for n in range(p):
            for m in range(q):
                dissip = eval(str(d_qdot[n,m]))
                #turn the dissip into tensor if it is int
                if(isinstance(dissip, int)):
                    dissip = torch.tensor(dissip)
                    Dissip[n,m,:] = dissip*Dissip[n,m,:]               
                else:
                    dissip = dissip.to(device).float()
                    # scales = torch.max(dissip) - torch.min(dissip)
                    # dissip = dissip/scales
                    Dissip[n,m,:] = dissip



        for n in range(j):
            for o in range(k):
                delta = eval(str(phi_q[n, o]))
                'time series of the value of phi_q'
                if(isinstance(delta, int)):
                    Delta[n, o, :] = delta*Delta[n, o, :]
                else:
                    delta = delta.to(device).float()
                    # scales = torch.max(delta) - torch.min(delta)
                    # delta = delta/scales
                    Delta[n, o, :] = delta
                # else:
                #     # Feature Scaling
                #     if(scaling == True):
                #         scales = torch.max(delta) - torch.min(delta)
                #         delta = delta/scales
                #     Delta[n, o, :] = delta

        for m in range(i):
            for n in range(j):
                for o in range(k):
                    zeta = eval(str(phi_qdot2[m, n, o]))
                    eta = eval(str(phi_qdotq[m, n, o]))
                    if(isinstance(zeta, int)):
                        zeta = torch.tensor(zeta)
                        Zeta[m, n, o, :] = zeta*Zeta[m, n, o, :]
                    else:
                        zeta = zeta.to(device).float()
                        # scales = torch.max(zeta) - torch.min(zeta)
                        # zeta = zeta/scales
                        Zeta[m, n, o, :] = zeta
                    if(isinstance(eta, int)):
                        eta = torch.tensor(eta)
                        Eta[m, n, o, :] = eta*Eta[m, n, o, :]
                    else:
                        eta = eta.to(device).float()
                        # scales = torch.max(eta) - torch.min(eta)
                        # eta = eta/scales
                        Eta[m, n, o, :] = eta
                    # if(isinstance(zeta, int)):

                    # else:
                    #     # Feature Scaling
                    #     if(scaling == True):
                    #         scales = torch.max(zeta) - torch.min(zeta)
                    #         zeta = zeta/scales
                    #     Zeta[m, n, o, :] = zeta

                    # if(isinstance(eta, int)):

                    # else:
                    #     # Feature Scaling
                    #     if(scaling == True):
                    #         scales = torch.max(eta) - torch.min(eta)
                    #         eta = eta/scales
                    #     Eta[m, n, o, :] = eta
    return Zeta, Eta, Delta, Dissip


def lagrangianforward(coef,coef_d, Zeta, Eta, Delta,Dissip, xdot, device):
    """
    Computing time series of q_tt (q double dot) prediction
    #Params:
    coef        : Coefficient corresponding to each basis function
    mask        : filter for coefficient below a certain threshold
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  
    """


    DL_q = torch.einsum('jkl,k->jl', Delta, coef)
    DL_qdot2 = torch.einsum('ijkl,k->ijl', Zeta, coef)
    DL_qdotq = torch.einsum('ijkl,k->ijl', Eta, coef)
    D_qdot = torch.einsum('jkl,k->jl', Dissip, coef_d)
    

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device).float()
    #xdot.requires_grad=True
    n = xdot.shape[1]
    q_t = xdot[:, :n//2].T
    q_tt = xdot[:, n//2:].T
    C = torch.einsum('ijl,il->jl', DL_qdotq, q_t)
    B = DL_q
    D = D_qdot
    A = torch.einsum('ijl,il->jl', DL_qdot2, q_tt)
    #A = torch.einsum('ijl->lij', DL_qdot2)
    #invA = torch.linalg.pinv(A)
    #invA = torch.einsum('lij->ijl', invA)
    loss = D-B+C+A
    #loss = q_tt - torch.einsum('ijl,jl->il', invA, B-C)
    # DL_q = torch.einsum('jkl,k->jl', Delta, coef)
    # DL_qdot2 = torch.einsum('ijkl,k->ijl', Zeta, coef)
    # DL_qdotq = torch.einsum('ijkl,k->ijl', Eta, coef)

    # if(torch.is_tensor(xdot) == False):
    #     xdot = torch.from_numpy(xdot).to(device).float()
    # q_t = xdot[:, :2].T
    # '2 might be a specific number'

    # C = torch.einsum('ijl,il->jl', DL_qdotq, q_t)
    # B = DL_q
    # A = torch.einsum('ijl->lij', DL_qdot2)
    # invA = torch.linalg.pinv(A)
    # invA = torch.einsum('lij->ijl', invA)
    # q_tt = torch.einsum('ijl,jl->il', invA, B-C)
    #loss is in shape (2,time-series). the first row is the loss of x0 and the second row is the loss of x1
    return loss






def ELforward(coef, Zeta, Eta, Delta, xdot, device):
    """
    Computing time series of total sum of Euler-Lagrange equation
    #Params:
    coef        : Coefficient corresponding to each basis function
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  

    #Returns:
    El          : Time series of the left hand side of Euler's Lagranges equation (n, time-series)
    """
    weight = coef
    DL_q = torch.einsum('jkl,k->jl', Delta, weight)
    DL_qdot2 = torch.einsum('ijkl,k->ijl', Zeta, weight)
    DL_qdotq = torch.einsum('ijkl,k->ijl', Eta, weight)
    n = xdot.shape[1]

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device).float()
    q_t = xdot[:, :n//2].T
    q_tt = xdot[:, n//2:].T

    C = torch.einsum('ijl,il->jl', DL_qdotq, q_t)
    B = DL_q
    A = torch.einsum('ijl,il->jl', DL_qdot2, q_tt)
    EL = A + C - B
    return EL


def Upsilonforward(Zeta, Eta, Delta, xdot, device):
    """
    Computing time series of total sum of Euler-Lagrange equation
    #Params:
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  

    #Returns:
    Upsilon          : Time series of the left hand side of Euler's Lagranges equation before multiplied with weight (n, time-series)
    """
    n = xdot.shape[1]

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device).float()
    q_t = xdot[:, :n//2].T
    q_tt = xdot[:, n//2:].T

    A = torch.einsum('ijkl,il->jkl', Zeta, q_tt)
    B = torch.einsum('ijkl,il->jkl', Eta, q_t)
    C = Delta

    Upsilon = A + B - C
    return Upsilon


def tauforward(coef, Zeta, Eta, Delta, xdot, device):
    '''
    Computing time series of tau (external input) prediction
    #Params:
    coef        : Coefficient corresponding to each basis function
    mask        : filter for coefficient below a certain threshold
    Zeta        : time-series of double derivative of basis functions w.r.t qdot and qdot  
    Eta         : time-series of double derivative of basis functions w.r.t qdot and q 
    Delta       : time-series of derivative of basis functions w.r.t q 
    xdot        : Time-series of states_dot data  
    '''
    weight = coef
    DL_q = torch.einsum('jkl,k->jl', Delta, weight)
    DL_qdot2 = torch.einsum('ijkl,k->ijl', Zeta, weight)
    DL_qdotq = torch.einsum('ijkl,k->ijl', Eta, weight)

    if(torch.is_tensor(xdot) == False):
        xdot = torch.from_numpy(xdot).to(device).float()
    q_t = xdot[:, :2].T
    q_tt = xdot[:, 2:].T

    C = torch.einsum('ijl,il->jl', DL_qdotq, q_t)
    B = DL_q
    A = torch.einsum('ijl,il->jl', DL_qdot2, q_tt)
    tau = A + C - B
    return tau


def SymGradient(func_description, q):
    '''
    Symbolic gradient of list of basis function w.r.t quantity q where q is subset of the states (can be position, velocity or acceleration)
    #Params:
    func_description    : list of basis functions (str) (d,)
    q                   : list of a quantity subset of the states (str) (d,)

    #Retuns:
    dfunc_dq                : gradient of basis functions w.r.t q (sympy matrix) (d,n)
    '''
    q = sympy.Matrix(q)
    func_description = sympy.Matrix(func_description)
    dfunc_dq = simplify(func_description.jacobian(q))
    return dfunc_dq


def TimeDerivativeSym(func_description, states, states_dot):
    '''
    Symbolic time derivative of basis function

    #Params:
    func_description     : list basis functions (str) (d,)
    states               : list states variable description (str) (n,)
    states_dot           : time derivative state_variable (str) (n,)

    #Return
    dfunc_dt             : symbolic time derivative of basis functions list (sympy matrix) (d,)
    '''
    func = sympy.Matrix(func_description)
    x = sympy.Matrix(states)
    x_dot = sympy.Matrix(states_dot)
    grad = func.jacobian(x)
    dfunc_dt = grad*x_dot
    return dfunc_dt


def TimeDerivativeSymGradient(gradfunc_description, states, states_dot):
    '''
    Symbolic time derivative of gradient of basis function w.r.t. quantity q which is a subset of the states

    #Params:
    gradfunc_description : gradient of basis function w.r.t. quantity q (sympy matrix) (d,n)
    states               : list states variable description (str) (n,)
    states_dot           : time derivative state_variable (str) (n,)


    #Return
    dgradfunc_description_dt : Symbolic time derivative of gradient of basis function w.r.t. quantity q (sympy matrix) (d,n)
    '''
    x = sympy.Matrix(states)
    x_dot = sympy.Matrix(states_dot)

    temp = gradfunc_description[:, 0].jacobian(x)*x_dot
    'row i = time derivative of basis function w.r.t q(i)'
    for i in range(1, len(states)//2):
        temp = temp.row_join(gradfunc_description[:, i].jacobian(x)*x_dot)
    dgradfunc_description_dt = temp
    return dgradfunc_description_dt


def SymVectorFuncSumOverStates(matrix_func):
    '''
    Sum of gradient of symbolic basis function over states
    #Params
    matrix_fun : gradient of symbolic basis function (sympy matrix) (d,n)

    #Return
    Sigma      : sum of gradeitn of symbolic basis function over states (sympy matrix) (d) 
    '''

    p, m = matrix_func.shape
    sigma = matrix_func[:, 0]
    for i in range(1, m):
        sigma += matrix_func[:, i]
    return sigma


def timeDerivativeLibraryMatrix(x, xdot, function_description, states, states_dot):
    """
    #Params:
    x                       : values of state variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    xdot                    : values of states_dot variables in torch tensor. In [x_dot,x_doubledot] format. Each row presents states at one time
    function_description    : list of basis functions (str) (d,)
    states                  : list states variable description (str) (n,)
    states_dot              : time derivative state_variable (str) (n,)

    #Return:
    time-series of time-derivative functions in torch.tensor
    """
    df_dt = TimeDerivativeSym(function_description, states, states_dot)
    df_dt = [str(f) for f in df_dt]
    from torch import cos, sin
    if((torch.is_tensor(x) == False) or (torch.is_tensor(xdot) == False)):
        x = torch.from_numpy(x)
        xdot = torch.from_numpy(xdot)

    column = []
    n = len(states)
    # Assign data to states and states dot
    for j in range(n):
        locals()[states[j]] = x[:, j]
        locals()[states_dot[j]] = xdot[:, j]
    # evaluate each function in function expression with data
    for func in df_dt:
        column.append(eval(func))
        'eval(func) returns a tensor with value of the function in time-series'
        'column: a list with d tensor inside, each tensor has t elements. t is the length of time'
    column = torch.stack(column)
    'column: a tensor with d*t elements'
    column = column.T
    return column


def LibraryMatrix(x, function_description, states, scaling=True):
    """
    #Params:
    x                       : values of variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    function_description    : list of basis functions (str) (d,)
    states                  : symbolic states' names (str)

    #Return:
    time-serie of calculated functions in torch.tensor
    """

    from torch import cos, sin
    if(torch.is_tensor(x) == False):
        x = torch.from_numpy(x)

    column = []
    n = len(states)
    # Assign data to data_description (states)
    for j in range(n):
        locals()[states[j]] = x[:, j]
    # evaluate each function in function expression with data
    for func in function_description:
        k = eval(func)
        if(isinstance(k, int)):
            column.append(k*torch.ones(x.shape[0]))
        else:
            # Feature Scaling
            if(scaling == True):
                scales = torch.max(k) - torch.min(k)
                k = k/scales
            column.append(k)
    column = torch.stack(column)
    column = column.T
    return column


def timeDerivativeLibraryTensor(x, xdot, matrix_func, states, states_dot):
    """
    #Params:
    x                       : values of state variables in torch tensor. In [x,x_dot] format. Each row presents states at one time
    xdot                    : values of states_dot variables in torch tensor. In [x_dot,x_doubledot] format. Each row presents states at one time
    matrix_func             : matrix of basis functions (str) (d,n) this n is half of the length of states!!!
    states                  : list states variable description (str) (n,)
    states_dot              : time derivative state_variable (str) (n,)

    #Return:
    time-series of time-derivative functions in torch.tensor
    """
    from torch import cos, sin
    if((torch.is_tensor(x) == False) or (torch.is_tensor(xdot) == False)):
        x = torch.from_numpy(x)
        xdot = torch.from_numpy(xdot)

    d, n = matrix_func.shape[0], len(states)
    b = x.shape[0]
    Eta = torch.ones(d, n//2, b)

    # Assign data to states and states dot
    for j in range(n):
        locals()[states[j]] = x[:, j]
        locals()[states_dot[j]] = xdot[:, j]

    # evaluate each function in function expression with data
    for i in range(matrix_func.shape[0]):
        for j in range(matrix_func.shape[1]):
            k = eval(str(matrix_func[i, j]))
            if(isinstance(k, int)):
                Eta[i, j, :] = k*Eta[i, j, :]
            else:
                Eta[i, j, :] = k
    return Eta
