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

def energy_size_correction(Lx, Ly, n_max, gamma, density, sk_over_k):
    """ Energy correction per particle \delta E_N """
    ang = 1e-10  # [m]
    atomic_unit_mass = 1.66053906660 * 10**-27  # [kg]
    m_he4 = 4.00260325413 * atomic_unit_mass  # [kg]
    kb = 1.380649 * 10**-23  # [J/K]
    hbar = 1.054571817 * 10**-34  # [Js]

    alpha = Lx / Ly  # [dimensionless]
    integral_term = np.sqrt(np.pi) * ellipe(1-alpha**2) / gamma**(3/2)

    one_dim_n_axis = np.arange(-n_max, n_max+1, 1) 
    nx, ny = np.meshgrid(one_dim_n_axis, one_dim_n_axis, indexing='ij')
    sum_term = np.sum(np.exp(-gamma * (nx**2 + ny**2)) * np.sqrt(nx**2 + alpha**2 * ny**2))

    diff = 2 * np.pi * (integral_term - sum_term) / (Lx**2 * Ly) 
    prefactor = hbar**2 / (4 * m_he4 * density * sk_over_k) / ang**2 # [J]
    return prefactor * diff / kb  # [K]


#################################################################################################
#         Plot S(k)/k vs k for multiple number densities to extrapolate S(k)/k for k->0         #
#                                              and                                              # 
#                get the energy correction \delta E_N for each number density                   #
#################################################################################################


for N in N_axis:
    N_key = f'N={N}'
    path_to_data = f'structure_factor_data/N={N}/'
    folders = sorted([folder for folder in os.listdir(path_to_data) if folder.startswith('density=')])

    densities = []
    s_of_k_over_k_at_0s = []

    for folder in folders:

        density = extract_density(folder)
        npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]

        ## Average the structure factor data over more than 1 file 
        bootstrap_samples = []
        for filename in npz_files:
            data = np.load(os.path.join(path_to_data, folder, filename))

            ## Load data
            n_max = data['n_max']
            ks = data['ks']/rm  # [A^-1]
            estimators = data['estimators']

            ## S(|k|) vs |k|
            ks_norm = jnp.linalg.norm(ks, axis=-1)
            mask = jnp.argsort(ks_norm)
            ks_norm_sorted = ks_norm[mask]
            ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
            struc_fac_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
            struc_fac_averaged = jnp.array([jnp.mean(x) for x in struc_fac_splitted])

            # Avoid |k|=0
            k = ks_norm_unique[1:]
            s = struc_fac_averaged[1:]
            bootstrap_samples.append(s)

        s = np.mean(np.array(bootstrap_samples), axis=0)  # average over all bootstrap samples (num_ks,)

        # Filter data points up to x-values of 0.7, perform spline fit
        k_max_for_fit = 1.4  # [A^-1]
        mask = k <= k_max_for_fit
        k_fit_data = k[mask]
        s_fit_data = s[mask]
        spline = UnivariateSpline(k_fit_data, s_fit_data/k_fit_data, s=2)
        k_fit_axis = np.linspace(0., k_fit_data.max(), 500)
        s_fit_values = spline(k_fit_axis)
        ####### plt.plot(k_fit_axis, s_fit_values, color='red', alpha=0.3, zorder=-1)# label='Spline fit', )
        s_of_k_over_k_at_0 = spline(0)
        print(f'density = {density}, lim_k->0 S(k)/k = {s_of_k_over_k_at_0:.5f}')

        ## Plot S(k) vs k
        ####### plt.scatter(k, s/k) 


        ## Get the energy correction \delta E_N
        Mx = Mxs[N_key]
        My = Mys[N_key]
        assert N == 2*Mx*My
        a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
        lx = a  # [A]
        ly = np.sqrt(3)*a  # [A]
        Lx = Mx*lx  # [A]
        Ly = My*ly  # [A]
        n_max = 500  # to define a grid size for the sum term
        gamma = 0.001  # exponential decay factor (lim \gamma -> 0)
        delta_E = energy_size_correction(Lx, Ly, n_max, gamma, density, s_of_k_over_k_at_0)

        ## Save the data in a dictionary
        density_key = f'n={density:.3f}'
        if density_key not in res:
            res[N_key][density_key] = {}
        res[N_key][density_key]['s_of_k_over_k_at_0'] = s_of_k_over_k_at_0
        res[N_key][density_key]['delta_E'] = delta_E

    ####### plt.xlim(0, 1.5)
    ####### plt.ylim(0, 0.7)
    ####### plt.show()






###########################################################################################
#      Plot the corrected/extrapolated energy, i.e. E+\delta E, vs number density n       #
###########################################################################################

working_dir = os.getcwd()
alpha = 1.37643  # hbar^2/(m*r_m^2*k_B) [K]
ns = 1024  # number of values to consider for the average
potential_type_subfolder = 'shifted_truncated_pot' # 'full_periodized_pot' 

P_c_extrapolated = []

for N in N_axis:
    print(f'N={N}')
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

    ## Select only the (density, energy) pairs for which we have the energy correction
    filtered_densities = []
    filtered_energies = []
    for density, energy in zip(densities, energies):
        density_key = f'n={density:.3f}'
        if density_key in res[N_key]:
            filtered_densities.append(density)
            filtered_energies.append(energy)
    densities = np.array(filtered_densities)
    energies = np.array(filtered_energies)

    ## Plot the energy and the extrapolated energy
    plt.scatter(1/densities, energies, label=f'$E/N$ [K] for $N={N}$')
    delta_E_s = np.array([res[N_key][f'n={density:.3f}']['delta_E'] for density in densities])
    # print(list(zip(densities,delta_E_s)))
    extrapolated_energies = energies + delta_E_s
    plt.scatter(1/densities, extrapolated_energies, label=f'$E/N+\delta E$ [K] for $N={N}$')

    ## Take two points in the vicinity of the coexistence region to calculate the slope between them
    n1 = 0.068
    n2 = 0.072
    y1 = energies[np.where(np.isclose(densities, n1))]
    y2 = energies[np.where(np.isclose(densities, n2))]
    slope = (y2-y1)/(1/n2-1/n1)
    print(f'slope with plain energies = {slope} KA^-2')
    y1 = extrapolated_energies[np.where(np.isclose(densities, n1))]
    y2 = extrapolated_energies[np.where(np.isclose(densities, n2))]
    slope = (y2-y1)/(1/n2-1/n1)
    print(f'slope with extrapolated energies = {slope} KA^-2')
    P_c_extrapolated.append(slope[0])

    ## Add a line in the coexistence region (using the extrapolated energies)
    plt.plot([1/n1, 1/n2], [y1, y2], color='k', linestyle='dashed', zorder=20)


diff = np.diff(np.array(P_c_extrapolated))
print(diff)

plt.legend()
plt.show()