import optuna
import numpy as np
from sklearn.metrics import accuracy_score
from double_pendulum_train_active_Adam import main as run_algorithm

# Define the success criterion
def is_success(real_coeff_values, estimated_coeff_values, tolerance=0.05):
    relative_errors = np.abs((np.array(real_coeff_values) - np.array(estimated_coeff_values)) / np.array(real_coeff_values))
    return np.all(relative_errors < tolerance)

# Define the objective function for the hyperparameter tuning
def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 1e-2)
    num_epochs = trial.suggest_int('num_epochs', 50, 500)
    num_epoch0 = trial.suggest_int('num_epoch0', 50, 500)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    tau = trial.suggest_categorical('tau', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    num_samples = trial.suggest_int('num_samples', 10, 200)
    lambda0 = trial.suggest_float('lambda0', 1e-3, 5e-1)
    lambda1 = trial.suggest_float('lambda1', 1e-3, 5e-1)
    lr_step = trial.suggest_float('lr_step', 5e-5, 1e-2)
    try:
        # Run your algorithm with the given hyperparameters
        param = {}
        param['L1'] = 1
        param['L2'] = 1
        param['m1'] = 1
        param['m2'] = 1
        param['b1'] = 0.5
        param['b2'] = 0.5
        param['tau0'] = tau
        param['omega1'] = 0.5
        param['omega2'] = 0.3
        param['phi'] = 0
        param['g'] = 9.81
        estimated_coeff_dict,relative_error = run_algorithm(param,num_sample=num_samples,Epoch=num_epochs,Epoch0=num_epoch0,lr=learning_rate,batch_size=batch_size,lam0=lambda0,lam=lambda1,lr_step=lr_step)

        # Calculate the success rate
        success = relative_error 
    except Exception as e:
        print(f"An error occurred while running the algorithm: {e}")
        success = 100  # Assign a low success rate for this trial

    return success

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# Print the best hyperparameters
print(study.best_params)
#save the best parameters and the success rate tendency
import pickle
with open('best_params_Adam.pkl', 'wb') as f:
    pickle.dump(study.best_params, f)
with open('success_rate_Adam.pkl', 'wb') as f:
    pickle.dump(study.trials_dataframe(), f)

