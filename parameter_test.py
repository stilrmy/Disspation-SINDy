import subprocess
import time
import os
import torch
print(torch.__version__)
print(torch.cuda.is_available())
env=os.environ.copy()

# Define sets of values for each argument
lr_values = [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
rho_values = [1e-3,1e-2, 0.5, 1.0, 2.0]
mu_values = [0.1, 0.01, 0.001,0.0001,0.00001]
gam_values = [1e-5,1e-4, 1e-3, 1e-2, 0.1]
epsilon_values = [1e-2,1e-3, 1e-4, 1e-5]

env['PATH'] = '/mnt/ssd1/stilrmy/anaconda3/envs/pytorch/bin:' + env['PATH']
subprocess.run("python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'", shell=True,env=env)

# Open a file to write the log
with open('performance_log.txt', 'w') as log_file:
    # Loop over all combinations of values
    for lr in lr_values:
        for rho in rho_values:
            for mu in mu_values:
                for gam in gam_values:
                    for epsilon in epsilon_values:
                        args = f"--lr {lr} --rho {rho} --mu {mu} --gam {gam} --epsilon {epsilon}"
                        start_time = time.time()
                        subprocess.run(f"python /mnt/ssd1/stilrmy/finished_work/SINDy_with_Rdf/sucess_rate_test.py {args}", shell=True,env=env)
                        end_time = time.time()
                        time_taken = end_time - start_time
                        log_message = f"Time for args {args}: {time_taken} seconds\n"
                        print(log_message, end='')
                        log_file.write(log_message)
