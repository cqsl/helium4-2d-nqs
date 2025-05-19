import os, re
import numpy as np


## This file is used to convert the swap estimator data to entropy data.
## However, the swap estimator data is not provided as it takes a lot of memory (many tens of gigabytes).


##  We assume R_axis is the same for all files in a folder

paths = [
    'entropy_data/N=30/Rn12=2.39_changing_origins/',
    'entropy_data/N=80/Rn12=2.39_changing_origins/',
    'entropy_data/N=80/Rn12=3.34_changing_origins/',
]


for i, path_to_data in enumerate(paths):
    density_folders = [folder for folder in os.listdir(path_to_data) if folder.startswith('density=')]
    Rn12 = float(re.search(r'Rn12=([\d.]+)', path_to_data).group(1))
    N = int(re.search(r'N=(\d+)', path_to_data).group(1))

    densities = []
    entropies = []
    entropies_err = []

    for folder in density_folders:
        density = float(re.search(r'density=([\d.]+)', folder).group(1))

        npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]
        
        bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)
        
        for file in npz_files:
            data = np.load(os.path.join(path_to_data, folder, file))
            R_axis = data['R_axis']  # (num_R=1,)
            estimators = data['estimators']  # (num_R,)
            s2 = -np.log(np.real(estimators))  # (num_R,)
            bootstrap_samples.append(s2)  # add one bootstrap sample

        bootstrapped_entropy = np.mean(bootstrap_samples, axis=0)  # (num_R,)
        bootstrapped_entropy_err = np.std(bootstrap_samples, axis=0) / np.sqrt(len(npz_files)) # (num_R,)

        ## Save mean and error to a .npz file 
        output_file = os.path.join(path_to_data, f'entropy_results_density={density}.npz')
        print(output_file)
        # np.savez(
        #     output_file, 
        #     density=density,
        #     R_axis=R_axis,
        #     entropy=bootstrapped_entropy,
        #     entropy_err=bootstrapped_entropy_err,
        # )
