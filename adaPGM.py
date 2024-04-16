import numpy as np
from numpy.linalg import norm
from numpy import dot
class OurRule:
    def __init__(self, gamma=0, t=1.1, norm_A=0, delta=1e-6, Theta=1.2):
        if gamma > 0:
            self.gamma = gamma
        elif norm_A > 0:
            self.gamma = 1 / (2 * Theta * t * norm_A)
        else:
            raise ValueError("You must provide gamma > 0 if norm_A = 0")
        self.t = t
        self.norm_A = norm_A
        self.delta = delta
        self.Theta = Theta
#define a class called Zero
class Zero:
    def __call__(self, x):
        return 0
    
class IndZero:
    def __call__(self, x):
        if x == 0:
            return 0
        else:
            return float('inf')


class NormL1:
    def __init__(self, lambda_=1):
        # Check if lambda_ is neither a scalar nor a numpy array (or list), raise an error
        if not isinstance(lambda_, (int, float, np.ndarray, list)):
            raise ValueError("λ must be real and nonnegative")
        # If lambda_ is a list, convert it to a numpy array for consistency
        if isinstance(lambda_, list):
            lambda_ = np.array(lambda_)
        # If lambda_ is an array, check if any element is negative
        if isinstance(lambda_, np.ndarray) and np.any(lambda_ < 0):
            raise ValueError("λ must be nonnegative")
        # If lambda_ is a scalar, check if it is negative
        if isinstance(lambda_, (int, float)) and lambda_ < 0:
            raise ValueError("λ must be nonnegative")
        
        self.lambda_ = lambda_
    
    def __call__(self, x):
        # Calculate the L1 norm, weighted by lambda
        x = np.asarray(x)  # Ensure x is a numpy array
        if isinstance(self.lambda_, np.ndarray):
            # If lambda_ is an array, perform element-wise multiplication
            return np.linalg.norm(self.lambda_ * x, 1)
        else:
            # If lambda_ is a scalar, multiply after calculating the norm
            return self.lambda_ * np.linalg.norm(x, 1)


def nan_to_zero(v):
    return np.where(np.isnan(v), 0, v)

def stepsize0(rule):
    gamma = rule.gamma
    sigma = rule.gamma * rule.t**2
    return gamma, sigma, (gamma, sigma)

def stepsize(rule, state, x1, grad_x1, x0, grad_x0):
    gamma1, gamma0 = state
    xi =  rule.t**2 * gamma1 * rule.norm_A**2
    C = norm(grad_x1 - grad_x0)**2/dot(grad_x1 - grad_x0, x1 - x0)
    C = nan_to_zero(C)
    L = dot(grad_x1 - grad_x0, x1 - x0)/norm(x1 - x0)**2
    L = nan_to_zero(L)
    D = gamma1 * L * (gamma1 * C - 1)
    epsilon = 1e-20  # small constant to prevent division by zero

    gamma = min(gamma1 * np.sqrt(1+gamma1/gamma0), 
                1/(2 * (rule.Theta + epsilon) * (rule.t + epsilon) * (rule.norm_A + epsilon)), 
                (gamma1 * np.sqrt(1 - 4 * xi * (1 + rule.delta)**2)/np.sqrt(2*(1 + rule.delta)*(D * np.sqrt(D**2 + xi * (1 - 4 * xi * (1 + rule.delta)**2))))))
    sigma = gamma * rule.t**2
    return gamma, sigma, (gamma, gamma1)

def prox(f, x, gamma=1):
    #flatten x
    x = x.flatten()
    y = np.zeros_like(x)
    if f.__class__ == Zero:
        y = x
        return y,0
    elif f.__class__ == IndZero:
        return y,0
    elif isinstance(f, NormL1):
        if isinstance(x, np.ndarray) and x.dtype == np.complex:
            if isinstance(gamma, np.ndarray):
                if isinstance(f.lambda_, np.ndarray):
                    assert len(y) == len(x) == len(f.lambda_) == len(gamma)
                    for i in range(len(x)):
                        gl = gamma[i] * f.lambda_[i]
                        y[i] = np.sign(x[i]) * (0 if abs(x[i]) <= gl else abs(x[i]) - gl)
                    return y,np.sum(f.lambda_ * np.abs(y))
                else:
                    assert len(y) == len(x) == len(gamma)
                    n1y = 0
                    for i in range(len(x)):
                        gl = gamma[i] * f.lambda_
                        y[i] = np.sign(x[i]) * (0 if abs(x[i]) <= gl else abs(x[i]) - gl)
                        n1y += np.abs(y[i])
                    return y, f.lambda_ * n1y
            else:
                if isinstance(f.lambda_, np.ndarray):
                    assert len(y) == len(x) == len(f.lambda_)
                    for i in range(len(x)):
                        gl = gamma * f.lambda_[i]
                        y[i] = np.sign(x[i]) * (0 if abs(x[i]) <= gl else abs(x[i]) - gl)
                    return y, np.sum(f.lambda_ *np.abs(y))
                else:
                    assert len(y) == len(x)
                    gl = gamma * f.lambda_
                    n1y = 0
                    for i in range(len(x)):
                        y[i] = np.sign(x[i]) * (0 if abs(x[i]) <= gl else abs(x[i]) - gl)
                        n1y += np.abs(y[i])
                    return y,f.lambda_ * n1y
        else:
            if isinstance(gamma, np.ndarray):
                if isinstance(f.lambda_, np.ndarray):
                    assert len(y) == len(x) == len(f.lambda_) == len(gamma)
                    for i in range(len(x)):
                        gl = gamma[i] * f.lambda_[i]
                        y[i] = x[i] + gl if x[i] <= -gl else (-gl if x[i] >= gl else -x[i])
                    return y,np.sum(f.lambda_ * np.abs(y))
                else:
                    assert len(y) == len(x) == len(gamma)
                    n1y = 0
                    for i in range(len(x)):
                        gl = gamma[i] * f.lambda_
                        y[i] = x[i] + gl if x[i] <= -gl else (-gl if x[i] >= gl else -x[i])
                        n1y += np.abs(y[i])
                    return y, f.lambda_ * n1y
            else:
                if isinstance(f.lambda_, np.ndarray):
                    assert len(y) == len(x) == len(f.lambda_)
                    for i in range(len(x)):
                        gl = gamma * f.lambda_[i]
                        y[i] = x[i] + gl if x[i] <= -gl else (-gl if x[i] >= gl else -x[i])
                    return y, np.sum(f.lambda_ * np.abs(y))
                else:
                    assert len(y) == len(x)
                    gl = gamma * f.lambda_
                    n1y = 0
                    for i in range(len(x)):
                        y[i] = x[i] + gl if x[i] <= -gl else (-gl if x[i] >= gl else -x[i])
                        n1y += np.abs(y[i])
                    return y,f.lambda_ * n1y
            
def convex_conjugate(f):
    if f.__class__ == Zero:
        return IndZero()
    if f.__class__ == IndZero:
        return Zero()
    return f

def eval_with_grad(f,w):
    res = np.dot(f.A,w) - f.b
    gradient = np.dot(np.transpose(f.A), res)
    return 0.5 * np.linalg.norm(res)**2, gradient

def adaptive_primal_dual(x,y,f,g,h,A,rule,tol=1e-5,max_it=10000):
    gamma,sigma,state = stepsize0(rule)
    h_conj = convex_conjugate(h)
    # 
    x_temp = np.repeat(x,2)
    A_x = np.dot(A,x_temp)
    _, grad_x = eval_with_grad(f,x_temp)
    y_temp = np.repeat(y,2)
    At_y = np.dot(np.transpose(A),y_temp)
    n = len(x)
    A_x = np.sum(A_x.reshape(n,2),axis=1)
    grad_x = np.sum(grad_x.reshape(n,2),axis=1)
    At_y = np.sum(At_y.reshape(n,2),axis=1)
    v = x - gamma * (grad_x + At_y)
    x_prev,A_x_prev,grad_x_prev = x,A_x,grad_x
    x,_ = prox(g,v,gamma)
    for it in range(max_it):
        x_temp = np.repeat(x,2)
        y_temp = np.repeat(y,2)
        A_x = np.dot(A,x_temp)
        f_x, grad_x = eval_with_grad(f,x_temp)
        n = len(x)
        A_x = np.sum(A_x.reshape(n,2),axis=1)
        grad_x = np.sum(grad_x.reshape(n,2),axis=1)
        primal_res = (v-x)/gamma + grad_x + At_y
        gamma_prev = gamma
        gamma,sigma,state = stepsize(rule,state,x,grad_x,x_prev,grad_x_prev)
        rho = gamma/gamma_prev
        w = y + sigma * ((1 + rho) * A_x - rho * A_x_prev)
        y,_ = prox(h_conj,w,sigma)
        dual_res = (w - y)/sigma - A_x
        norm_res = np.sqrt(norm(primal_res)**2 + norm(dual_res)**2)
        if norm_res <= tol:
            return x, y, it
        At_y = np.dot(np.transpose(A),y_temp)
        At_y = np.sum(At_y.reshape(n,2),axis=1)
        v = x - gamma * (grad_x + At_y)
        x_prev,A_x_prev,grad_x_prev = x,A_x,grad_x
        x, _ = prox(g,v,gamma)
        # print('Iteration: ', it, 'Norm of residual: ', norm_res,'coefficients: ', x)
        if it % 10 == 0:
            print('Iteration: ', it, 'Norm of residual: ', norm_res)
        # print('Iteration: ', it, 'Norm of residual: ', norm_res)
    return x, norm_res
