import numpy as np
import pandas as pd
from tqdm import tqdm
from single_pendulum_train_active_APGD import main as run_algorithm

# Define the success criterion
def is_success(error, tolerance=0.2):
    return error < tolerance

# Initialize lists to store results for each noise level
success_rates = []
average_errors = []
average_epochs = []

Lags = {}
noise_list = [0,1e-3,2e-2,6e-2,1e-1]

# Run the algorithm for each noise level
for noise_level in noise_list:
    errors = []
    epochs = []
    for i in tqdm(range(10)):
        try:
            Lag, error, epoch = run_algorithm(noiselevel=noise_level,display=False,device='cuda:1')
            if is_success(error):
                errors.append(error)
                epochs.append(epoch)
                Lags[str(noise_level)] = Lag
        except Exception as e:
            print(f"Error: {e}")

    # Record the results for this noise level
    average_errors.append(np.mean(errors))
    average_epochs.append(np.mean(epochs))

# Save the results to a file
with open('results.txt', 'w') as f:
    for noise_level, avg_error, avg_epoch in zip(noise_list, average_errors, average_epochs):
        f.write(f"Noise level: {noise_level}\n")
        f.write(f"Average error: {avg_error}\n")
        f.write(f"Average epochs: {avg_epoch}\n")
        f.write(f"Lag: {Lags[str(noise_level)]}\n")
        f.write("\n")