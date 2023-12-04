import sympy as sp
from sympy import sin, cos


# Define your symbols
theta1, theta2 = sp.symbols('theta1 theta2')
theta1_dot, theta2_dot = sp.symbols('theta1_dot theta2_dot')
theta1_dot_dot, theta2_dot_dot = sp.symbols('theta1_dot_dot theta2_dot_dot') # Time derivatives of theta1_dot and theta2_dot
m1, m2 = sp.symbols('m1 m2') # Masses
l1, l2 = sp.symbols('l1 l2') # Lengths
g = sp.symbols('g') # Gravity



# Define the dissipation function for viscous damping
b1, b2 = sp.symbols('b1 b2') # Damping coefficients
D = 0.5 * b1 * theta1_dot**2 + 0.5 * b2 * theta2_dot**2 # Dissipation function


# Define the Lagrangian L here
#L = 0.5*(m1+m2)*l1**2*theta1_dot**2 + 0.5*m2*l2**2*theta2_dot**2 + m2*l1*l2*theta1_dot*theta2_dot*cos(theta1)*cos(theta2)#+m2*l1*l2*theta1_dot*theta2_dot*sin(theta1)*sin(theta2) + (m1+m2)*g*l1*cos(theta1) + m2*g*l2*cos(theta2) 
# Kinetic energy (without variable length pendulums)
T = 0.5 * m1 * l1**2 * theta1_dot**2 + 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 2 * l1 * l2 * theta1_dot * theta2_dot * sp.cos(theta1 - theta2))

# Potential energy
U = - m1 * g * l1 * sp.cos(theta1) - m2 * g * (l1 * sp.cos(theta1) + l2 * sp.cos(theta2))

# Lagrangian
L = T - U


# Compute partial derivatives
dL_dtheta1 = sp.diff(L, theta1)
dL_dtheta2 = sp.diff(L, theta2)
dL_dtheta1_dot = sp.diff(L, theta1_dot)
dL_dtheta2_dot = sp.diff(L, theta2_dot)

# Compute time derivatives
#ddt_dL_dtheta1_dot = (m1+m2)*l1**2*theta1_dot_dot+m2*l1*l2*theta2_dot_dot*cos(theta1-theta2)-m2*l1*l2*theta2_dot*(theta1_dot-theta2_dot)*sin(theta1-theta2)
#ddt_dL_dtheta2_dot = (m1+m2)*l2**2*theta2_dot_dot+m2*l1*l2*theta1_dot_dot*cos(theta1-theta2)-m2*l1*l2*theta1_dot*(theta1_dot-theta2_dot)*sin(theta1-theta2)*(theta1_dot-theta2_dot)
# Compute time derivatives of the partial derivatives with respect to the velocities
ddt_dL_dtheta1_dot = sp.diff(dL_dtheta1_dot, theta1) * theta1_dot + sp.diff(dL_dtheta1_dot, theta2) * theta2_dot + sp.diff(dL_dtheta1_dot, theta1_dot) * theta1_dot_dot + sp.diff(dL_dtheta1_dot, theta2_dot) * theta2_dot_dot
ddt_dL_dtheta2_dot = sp.diff(dL_dtheta2_dot, theta1) * theta1_dot + sp.diff(dL_dtheta2_dot, theta2) * theta2_dot + sp.diff(dL_dtheta2_dot, theta1_dot) * theta1_dot_dot + sp.diff(dL_dtheta2_dot, theta2_dot) * theta2_dot_dot



# Compute non-conservative forces
Q_nc1 = -sp.diff(D, theta1_dot)
Q_nc2 = -sp.diff(D, theta2_dot)

# Euler-Lagrange equations
eq1 = sp.Eq(ddt_dL_dtheta1_dot - dL_dtheta1, Q_nc1)
eq2 = sp.Eq(ddt_dL_dtheta2_dot - dL_dtheta2, Q_nc2)

# Solve the equations for theta1_dot_dot and theta2_dot_dot
solution = sp.solve((eq1, eq2), (theta1_dot_dot, theta2_dot_dot))
print('theta1_dot_dot: ',solution[theta1_dot_dot])
print('theta2_dot_dot: ',solution[theta2_dot_dot])


# Print the results

