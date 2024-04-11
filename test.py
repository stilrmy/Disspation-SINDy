from sympy import expand, sympify, symbols, parse_expr, simplify,cos,sin
import numpy as np

expression_string = "9.579*cos(x1)+0.491*x1_t**2+19.246*cos(x0)+0.983*x0_t**2+0.983*x0_t*x1_t*cos(x0 - x1)"
#turn the string into a sympy expression
# exprs = parse_expr(expression_string)
exprs = simplify(expression_string)
coeff_dict = exprs.as_coefficients_dict()
print(coeff_dict)

x0_t, x1_t, x0, x1 = symbols('x0_t x1_t x0 x1')

param = {}
param['L1'] = 1
param['L2'] = 1
param['m1'] = 1
param['m2'] = 1
param['g'] = 9.81

expr = ['x0_t**2', 'x1_t**2', 'x0_t*x1_t*cos(x0 - x1)', 'cos(x0)', 'cos(x1)']
xi_Lcpu = np.array([0.983, 0.491, 0.983, 19.246, 9.579]) 

m1,m2,l1,l2,g = param['m1'],param['m2'],param['L1'],param['L2'],param['g']
# Define the symbols
x0, x1, x0_t, x1_t = symbols('x0 x1 x0_t x1_t')

# Define the real Lagrangian model
L_real = m1*l1**2*x0_t**2/2 + m2*(l1**2*x0_t**2/2 + l2**2*x1_t**2/2 + l1*l2*x0_t*x1_t*cos(x0)*cos(x1)+l1*l2*x0_t*x1_t*sin(x0)*sin(x1)) + (m1+m2)*g*l1*cos(x0) + m2*g*l2*cos(x1)

# Simplify the real Lagrangian model if x0_t*x1_t*cos(x0 - x1) appears in the estimated candidates
if 'x0_t*x1_t*cos(x0 - x1)' in expr:
    L_real_simplified = simplify(L_real)
    print(L_real_simplified)
else:
    L_real_simplified = L_real

# Get the real coefficients
real_coeff_dict = L_real_simplified.as_coefficients_dict()
real_coeff_dict = {str(key): val for key, val in real_coeff_dict.items()}
# Create a dictionary of estimated coefficients
estimated_coeff_dict = dict(zip(expr, xi_Lcpu))

#scale the x0_t**2 and use that scaler to scale the other coefficients
scale = real_coeff_dict['x0_t**2']/estimated_coeff_dict['x0_t**2']

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

for real, estimated in zip(real_coeff_values, estimated_coeff_values):
    # Avoid division by zero
    if real != 0:
        sum_relative_errors += (real - estimated) / real
    else:
        sum_relative_errors += float('inf') if estimated != 0 else 0

# Print the relative errors
print("The relative errors are:", sum_relative_errors)



#coef_dict = simplify(sympify(expression_string)).as_coefficients_dict()
# coef_dict = simplify(expression_string).as_coefficients_dict()

# threshold = 0.1

# # Filtering the dictionary
# filtered_candidates = {key: val for key, val in coef_dict.items() if val >= threshold}
# a = list(filtered_candidates.keys())
# c = []
# for can in a:
#     c.append('{}'.format(can))
# print(c)
# b = ['x0_t**2','x1_t**2']
# b = np.array(b)
# print(b)
# omega = 0.5*np.pi
# print(np.cos(omega*np.arange(0,5,0.01)))
# import matplotlib.pyplot as plt

# # Generate the data
# x = np.arange(0, 5, 0.01)
# y = np.cos(omega * x)

# # Create the plot
# plt.figure()
# plt.plot(x, y)
# plt.title('Cosine Plot')
# plt.xlabel('x')
# plt.ylabel('cos(omega*x)')

# # Save the plot as an image
# plt.savefig('/mnt/ssd1/stilrmy/finished_work/SINDy_with_Rdf/cosine_plot.png')

# # Show the plot
# plt.show()

