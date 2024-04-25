from sympy import expand, sympify, symbols, parse_expr, simplify,cos,sin
import numpy as np

# expression_string = "10.0*x0_t**2*sin(x1) + 4.352*x0_t**2*sin(x0 - x1) + 4.352*x0_t**2*sin(x0 + x1) + 2.802*x0_t*x1_t*sin(2*x0) + 3.993*x0_t*x1_t*sin(x1) - 2.896*x0_t*x1_t*sin(x0 - x1) + 2.896*x0_t*x1_t*sin(x0 + x1) + 2.133*x0_t*sin(x0) - 3.116*x0_t*sin(x0 - x1) + 3.116*x0_t*sin(x0 + x1) + 3.551*x0_t*cos(x0 - x1) - 0.044*x0_t*cos(x0 + x1) - 0.186*x1_t**2*sin(x0 - x1) + 0.186*x1_t**2*sin(x0 + x1) - 0.425*x1_t**2*cos(2*x0) + 5.174*x1_t**2 + 1.115*x1_t*sin(2*x0) + 4.212*x1_t*cos(x0 - x1) - 4.212*x1_t*cos(x0 + x1) - 3.648*cos(2*x1) + 1.055*cos(x0 - x1) + 1.055*cos(x0 + x1) + 3.648"
# #turn the string into a sympy expression
# # exprs = parse_expr(expression_string)
# exprs = simplify(expression_string)
# coeff_dict = exprs.as_coefficients_dict()
# print(coeff_dict)
# print(exprs)
x0_t, x1_t, x0, x1 = symbols('x0_t x1_t x0 x1')

param = {}
param['L'] = 1

param['M'] = 1
param['m'] = 0.5
param['g'] = 9.81



l,M,m,g = param['L'],param['M'],param['m'],param['g']
# Define the symbols
x0, x1, x0_t, x1_t = symbols('x0 x1 x0_t x1_t')

# Define the real Lagrangian model
L_real = 0.5*(M+m)*x1_t**2+m*l*x0_t*x1_t*cos(x0)+0.5*m*l**2*x0_t**2+m*g*l*cos(x0)

# Simplify the real Lagrangian model if x0_t*x1_t*cos(x0 - x1) appears in the estimated candidates


# Get the real coefficients
real_coeff_dict = L_real.as_coefficients_dict()
real_coeff_dict = {str(key): val for key, val in real_coeff_dict.items()}
print(real_coeff_dict)
# Create a dictionary of estimated coefficients
estimated_coeff_dict = {cos(x0): 20.0000000000000, x1_t**2: 2.88000000000000, x0_t*x1_t*cos(x0): 1.92600000000000}
print(estimated_coeff_dict)
estimated_coeff_dict['x0_t**2'] = 1
estimated_coeff_dict = {str(key): val for key, val in estimated_coeff_dict.items()}
#add the x0_t**2 term

print(estimated_coeff_dict)

#scale the x0_t**2 and use that scaler to scale the other coefficients
scale = real_coeff_dict['x0_t**2']/estimated_coeff_dict['x0_t**2']
print(scale)

for key in estimated_coeff_dict.keys():
    estimated_coeff_dict[key] = estimated_coeff_dict[key]*scale
print(estimated_coeff_dict)


# Calculate the relative error
# Initialize the sum of relative errors
sum_relative_errors = 0

for cand in estimated_coeff_dict.keys():
    #check if the term is in the real coefficients
    if cand in real_coeff_dict.keys():
        real_coeff = real_coeff_dict[cand]
        estimated_coeff = estimated_coeff_dict[cand]
        relative_error = abs(real_coeff - estimated_coeff) / abs(real_coeff)
        sum_relative_errors += relative_error
    else:
        print(f"The term {cand} is not in the real coefficients")
        sum_relative_errors += 1


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

