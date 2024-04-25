import numpy as np
import pandas as pd
from tqdm import tqdm
from cart_pendulum_train_active_PGD import main as run_algorithm

# Define the success criterion
def is_success(error, tolerance=0.2):
    return error < tolerance

# Define the noise levels to test
noise_levels = [0, 1e-3, 2e-2, 6e-2, 1e-1]

# Initialize lists to store results for each noise level
success_rates = []
average_errors = []
average_epochs = []

# Run the algorithm for each noise level
for noise_level in noise_levels:
    successes = 0
    total_runs = 100
    errors = []
    epochs = []

    for i in tqdm(range(total_runs), desc=f"Noise level: {noise_level}"):
        try:
            # Run your algorithm with the current noise level
            estimated_coeff_dict, error, epoch = run_algorithm(noiselevel=noise_level,display=False,device='cuda:3')

            # Record the error and the number of epochs
            if is_success(error):
                successes += 1
                errors.append(error)
                epochs.append(epoch)
        except Exception as e:
            print(f"An error occurred while running the algorithm: {e}")

    # Calculate the success rate
    success_rate = successes / total_runs

    # Record the results for this noise level
    success_rates.append(success_rate)
    average_errors.append(np.mean(errors))
    average_epochs.append(np.mean(epochs))

# Create a DataFrame to store the results
df = pd.DataFrame({
    'Noise Level': noise_levels,
    'Success Rate': success_rates,
    'Average Error': average_errors,
    'Average Epochs': average_epochs
})

# Save the DataFrame to a CSV file
df.to_csv('results.csv', index=False)

# Print the success rates, average errors, and average number of epochs for each noise level
for noise_level, success_rate, avg_error, avg_epochs in zip(noise_levels, success_rates, average_errors, average_epochs):
    print(f"Noise level: {noise_level}")
    print(f"Success rate: {success_rate*100}%")
    print(f"Average error: {avg_error}")
    print(f"Average number of epochs: {avg_epochs}")
    print()