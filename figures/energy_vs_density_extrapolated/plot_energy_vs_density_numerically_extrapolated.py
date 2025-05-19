import os, re, json
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline
from scipy.special import ellipe
import matplotlib.pyplot as plt


rm = 2.9673  # [A]

Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}

N_axis = [30,56,80]
res = {}  # to store all the data in there
for N in N_axis: res[f'N={N}'] = {}

def extract_density(filename): return float(re.search(r'density=([\d.]+)', filename).group(1))



###########################################################################################
#                                 Plot E/N vs n (vs N)                                    #
###########################################################################################

working_dir = os.getcwd()
alpha = 1.37643  # hbar^2/(m*r_m^2*k_B) [K]
ns = 1024  # number of values to consider for the average
potential_type_subfolder = 'shifted_truncated_pot' # 'full_periodized_pot' 


for N in N_axis:
    N_key = f'N={N}'
    data_path =  'log_files/' + potential_type_subfolder + f'/N={N}/'
    full_path = os.path.join(working_dir, data_path)

    def extract_density(filename):
        return float(re.search(r'density=([\d.]+)', filename).group(1))

    filenames = os.listdir(full_path)
    filenames = [filename for filename in filenames if filename.endswith('.log')]
    sorted_filenames = sorted(filenames, key=extract_density)

    densities = []
    energies = []
    energies_err = []

    for filename in sorted_filenames:
        if filename.startswith('he4') and filename.endswith('.log') and f'N={N}' in filename: # filename.startswith('_he4') and 

            ## Extract the data from the filename
            density = float(re.search(r'density=([\d.]+)', filename).group(1))

            ## Load the data contained in the file
            with open(full_path + filename) as f:
                data_log = json.load(f)
            energy = np.array(data_log['Energy']['Mean'])
            energy_err = np.array(data_log['Energy']['Sigma'])

            mean_energy = alpha * np.mean(energy[-ns:]) / N  # [K]
            mean_energy_err = alpha * np.mean(energy_err[-ns:]) / N

            ## Store the results in the appropriate list
            densities.append(density)
            energies.append(mean_energy)
            energies_err.append(mean_energy_err)

    densities = np.array(densities)
    energies = np.array(energies)

    ## Plot the energy and the extrapolated energy
    plt.scatter(1/densities, energies, label=f'$E/N$ [K] for $N={N}$')

plt.legend()
plt.show()





###########################################################################################
#                      Plot E/N vs N^{-3/2} (for a fixed density)                         #
###########################################################################################


working_dir = os.getcwd()
alpha = 1.37643  # hbar^2/(m*r_m^2*k_B) [K]
ns = 1024  # number of values to consider for the average
potential_type_subfolder = 'shifted_truncated_pot' # 'full_periodized_pot' 


n = 0.071  # fixed a number density [A^-2]
energies_at_n = [] 

for N in N_axis:
    N_key = f'N={N}'
    data_path =  'log_files/' + potential_type_subfolder + f'/N={N}/'
    full_path = os.path.join(working_dir, data_path)

    def extract_density(filename):
        return float(re.search(r'density=([\d.]+)', filename).group(1))

    filenames = os.listdir(full_path)
    filenames = [filename for filename in filenames if filename.endswith('.log') and extract_density(filename) == n]    
    sorted_filenames = sorted(filenames, key=extract_density)

    densities = []
    energies = []
    energies_err = []

    for filename in sorted_filenames:
        if filename.startswith('he4') and filename.endswith('.log') and f'N={N}' in filename: # filename.startswith('_he4') and 

            ## Extract the data from the filename
            density = float(re.search(r'density=([\d.]+)', filename).group(1))

            ## Load the data contained in the file
            with open(full_path + filename) as f:
                data_log = json.load(f)
            energy = np.array(data_log['Energy']['Mean'])
            energy_err = np.array(data_log['Energy']['Sigma'])

            mean_energy = alpha * np.mean(energy[-ns:]) / N  # [K]
            mean_energy_err = alpha * np.mean(energy_err[-ns:]) / N

            ## Store the results in the appropriate list
            densities.append(density)
            energies.append(mean_energy)
            energies_err.append(mean_energy_err)

    densities = np.array(densities)
    energies = np.array(energies)

    if density == n: 
        energies_at_n.append(energies)


x = np.array(N_axis) ** (-3/2)
y = np.array([x[0] for x in energies_at_n])

# Perform linear fit
fit_params = np.polyfit(x, y, 1)
slope, intercept = fit_params
E_inf = intercept  # extrapolated energy to N->infinity
print(f'E_inf = {E_inf:.5f} K')

delta_E = abs(E_inf - y)
print(N_axis, delta_E)

# Generate line values
x_fit = np.linspace(0, max(x), 100)
y_fit = slope * x_fit + intercept

# Plot the energy and the extrapolated energy
plt.scatter(x, y, label='Data points')
plt.plot(x_fit, y_fit, color='red', label='Linear fit')

plt.xlabel('$N^{-3/2}$')
plt.ylabel('$E/N$ [K]')
plt.legend()
plt.tight_layout()
plt.show()




## Manual estimate of \delta P = (\delta E_N^S - \delta E_N^L) / (n_S - n_L) for N=80 (using n=0.069 and n=0.071 values)
def get_volume_per_particle(N, density):
    Mx = Mxs[f'N={N}']
    My = Mys[f'N={N}']
    a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
    lx = a  # [A]
    ly = np.sqrt(3)*a  # [A]
    Lx = Mx*lx  # [A]
    Ly = My*ly  # [A]
    L = jnp.array([Lx,Ly])  # [A]
    return np.prod(L) / N

delta_E_s = 0.0246242
delta_E_l = 0.0202216
v_s = get_volume_per_particle(N=80, density=0.069)
v_l = get_volume_per_particle(N=80, density=0.071)

delta_P = (delta_E_s - delta_E_l) / (v_s - v_l)
print(f'delta_P = {delta_P:.5f} KA^{-2}')