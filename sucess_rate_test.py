import numpy as np
import pandas as pd
from tqdm import tqdm
from finished_work.SINDy_with_Rdf.double_pendulum_train_active_APGD import main as run_algorithm

# Define the success criterion
def is_success(error, tolerance=0.2):
    return error < tolerance

param = {}
param['L1'] = 1
param['L2'] = 1
param['m1'] = 1
param['m2'] = 1
param['b1'] = 0.2
param['b2'] = 0.2
param['tau0'] = 0.1
param['omega1'] = 0.5
param['omega2'] = 0.3
param['phi'] = 0
param['g'] = 9.81

param1 = param.copy()
param1['l1'] = 0.5
param1['l2'] = 0.5
param1['m1'] = 1.5
param1['m2'] = 1.5
param2 = param.copy()
param2['m1'] = 1.5
param2['m2'] = 1.5
param3 = param.copy()
param3['l1'] = 1.5
param3['l2'] = 1.5
param3['m1'] = 1.5
param3['m2'] = 1.5
param4 = param.copy()
param4['l1'] = 0.5
param4['l2'] = 0.5
param5 = param.copy()
param6 = param.copy()
param6['l1'] = 1.5
param6['l2'] = 1.5
param7 = param.copy()
param7['m1'] = 0.5
param7['m2'] = 0.5
param7['l1'] = 0.5
param7['l2'] = 0.5
param8 = param.copy()
param8['m1'] = 0.5
param8['m2'] = 0.5
param9 = param.copy()
param9['m1'] = 0.5
param9['m2'] = 0.5
param9['l1'] = 1.5
param9['l2'] = 1.5

param0 = {}
param0['L1'] = 1
param0['L2'] = 1
param0['m1'] = 1
param0['m2'] = 1
param0['b1'] = 0.05
param0['b2'] = 0.05
param0['tau0'] = 0.1
param0['omega1'] = 0.5
param0['omega2'] = 0.3
param0['phi'] = 0
param0['g'] = 9.81

param10 = param0.copy()
param10['l1'] = 0.5
param10['l2'] = 0.5
param10['m1'] = 1.5
param10['m2'] = 1.5
param11 = param0.copy()
param11['m1'] = 1.5
param11['m2'] = 1.5
param12 = param0.copy()
param12['l1'] = 1.5
param12['l2'] = 1.5
param12['m1'] = 1.5
param12['m2'] = 1.5
param13 = param0.copy()
param13['l1'] = 0.5
param13['l2'] = 0.5
param14 = param0.copy()
param15 = param0.copy()
param15['l1'] = 1.5
param15['l2'] = 1.5
param16 = param0.copy()
param16['m1'] = 0.5
param16['m2'] = 0.5
param16['l1'] = 0.5
param16['l2'] = 0.5
param17 = param0.copy()
param17['m1'] = 0.5
param17['m2'] = 0.5
param18 = param0.copy()
param18['m1'] = 0.5
param18['m2'] = 0.5
param18['l1'] = 1.5
param18['l2'] = 1.5

# Define the noise levels to test
param_list = [param1,param2,param3,param4,param5,param6,param7,param8,param9,param0,param10,param11,param12,param13,param14,param15,param16,param17,param18]

param_list_ = ['param1','param2','param3','param4','param5','param6','param7','param8','param9','param0','param10','param11','param12','param13','param14','param15','param16','param17','param18']

# Initialize lists to store results for each noise level
success_rates = []
average_errors = []
average_epochs = []

# Run the algorithm for each noise level
for params in param_list:
    errors = []
    epochs = []
    j = 0

    for i in tqdm(range(10)):
        try:
            error, epoch = run_algorithm(param=params,display=False,device='cuda:1')
            if is_success(error):
                print(f"Success! Error: {error}, Epoch: {epoch}")
                errors.append(error)
                epochs.append(epoch)
        except Exception as e:
            print(f"Error: {e}")
        

    j += 1
    # Calculate the success rate


    # Record the results for this noise level

    average_errors.append(np.mean(errors))
    average_epochs.append(np.mean(epochs))

# Create a DataFrame to store the results
df = pd.DataFrame({
    'param set': param_list_,
    'Average Error': average_errors,
    'Average Epochs': average_epochs
})

# Save the DataFrame to a CSV file
df.to_csv('/mnt/ssd1/stilrmy/finished_work/SINDy_with_Rdf/results_PGD_PT.csv', index=False)

# Print the success rates, average errors, and average number of epochs for each noise level
for param, success_rate, avg_error, avg_epochs in zip(param_list_, success_rates, average_errors, average_epochs):
    print(f"param set: {param}")
    print(f"Success rate: {success_rate*100}%")
    print(f"Average error: {avg_error}")
    print(f"Average number of epochs: {avg_epochs}")
    print()