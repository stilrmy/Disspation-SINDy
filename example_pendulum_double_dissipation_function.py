import sympy as sp
from sympy import sin, cos

# Define symbols
theta1, theta2, t = sp.symbols('theta1 theta2 t')
theta1_dot, theta2_dot = sp.symbols('theta1_dot theta2_dot')
theta1_dot_dot, theta2_dot_dot = sp.symbols('theta1_dot_dot theta2_dot_dot')  # Time derivatives
m1, m2 = sp.symbols('m1 m2')  # Masses
l1, l2 = sp.symbols('l1 l2')  # Lengths
g = sp.symbols('g')  # Gravity
tau0, omega1, omega2, phi = sp.symbols('tau0 omega1 omega2 phi')  # Symbols for sinusoidal torque

# Define the dissipation function for viscous damping
b1, b2 = sp.symbols('b1 b2')  # Damping coefficients
D = 0.23 * theta1_dot**2 + 0.21 * theta2_dot**2  # Dissipation function


# Lagrangian
L = 0.988*theta1_dot**2+0.488*theta2_dot**2+0.973*theta1_dot*theta2_dot*sp.sin(theta1)*sp.sin(theta2)+0.983*theta1_dot*theta2_dot*sp.cos(theta1)*sp.cos(theta2)+19.62*sp.cos(theta1)+9.688*sp.cos(theta2)
# Compute partial derivatives
dL_dtheta1 = sp.diff(L, theta1)
dL_dtheta2 = sp.diff(L, theta2)
dL_dtheta1_dot = sp.diff(L, theta1_dot)
dL_dtheta2_dot = sp.diff(L, theta2_dot)

# Compute time derivatives of the partial derivatives with respect to the velocities
ddt_dL_dtheta1_dot = sp.diff(dL_dtheta1_dot, theta1) * theta1_dot + sp.diff(dL_dtheta1_dot, theta2) * theta2_dot + sp.diff(dL_dtheta1_dot, theta1_dot) * theta1_dot_dot + sp.diff(dL_dtheta1_dot, theta2_dot) * theta2_dot_dot
ddt_dL_dtheta2_dot = sp.diff(dL_dtheta2_dot, theta1) * theta1_dot + sp.diff(dL_dtheta2_dot, theta2) * theta2_dot + sp.diff(dL_dtheta2_dot, theta1_dot) * theta1_dot_dot + sp.diff(dL_dtheta2_dot, theta2_dot) * theta2_dot_dot

# Non-conservative forces with sinusoidal torque on theta1
Q_nc1 = -sp.diff(D, theta1_dot)  # Sinusoidal torque added here
Q_nc2 = -sp.diff(D, theta2_dot)  # Sinusoidal torque added here

# Euler-Lagrange equations
eq1 = sp.Eq(ddt_dL_dtheta1_dot - dL_dtheta1, Q_nc1+tau0)
eq2 = sp.Eq(ddt_dL_dtheta2_dot - dL_dtheta2, Q_nc2)

# Solve the equations for theta1_dot_dot and theta2_dot_dot
solution = sp.solve((eq1, eq2), (theta1_dot_dot, theta2_dot_dot))

#do the simplifications
solution[theta1_dot_dot] = sp.simplify(solution[theta1_dot_dot])
solution[theta2_dot_dot] = sp.simplify(solution[theta2_dot_dot])

#replace the theta1 with theta, theta2 with x, theta1_dot with theta_t, theta2_dot with x_t
solution[theta1_dot_dot] = solution[theta1_dot_dot].subs({ theta1_dot: 'theta1_t', theta2_dot: 'theta2_t'})
solution[theta2_dot_dot] = solution[theta2_dot_dot].subs({ theta1_dot: 'theta1_t', theta2_dot: 'theta2_t'})

print('theta1_dot_dot: ', solution[theta1_dot_dot])
print('theta2_dot_dot: ', solution[theta2_dot_dot])
