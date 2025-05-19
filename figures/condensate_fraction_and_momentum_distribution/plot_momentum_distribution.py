import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import os 
import re


########################### PLOT SETTINGS ###########################
## The following is taken from: https://github.com/quantum-journal/quantum-journal/tree/v5.0
_widths = {
    'onecolumn': {
        'a4paper' : 5.93,  # a4paper columnwidth = 426.79135 pt = 5.93 in
        'letterpaper' : 6.16  # letterpaper columnwidth = 443.57848 pt = 6.16 in
    },
    'twocolumn': {
        'a4paper' : 3.22,  # a4paper columnwidth = 231.84843 pt = 3.22 in
        'letterpaper' : 3.34  # letterpaper columnwidth = 240.24199 pt = 3.34 in
    }
}
_wide_widths = {
    'onecolumn': {
        'a4paper' : 5.93,  # a4paper wide columnwidth = 426.79135 pt = 5.93 in
        'letterpaper' : 6.16  # letterpaper wide columnwidth = 443.57848 pt = 6.16 in
    },    
    'twocolumn': {
        'a4paper' : 6.72,  # a4paper wide linewidth = 483.69687 pt = 6.72 in
        'letterpaper' : 6.95  # letterpaper wide linewidth = 500.48400 pt = 6.95 in
    }
}
_fontsizes = {
    10 : {
        'tiny' : 5,
        'scriptsize' : 7,
        'footnotesize' : 8, 
        'small' : 9, 
        'normalsize' : 10,
        'large' : 12, 
        'Large' : 14, 
        'LARGE' : 17,
        'huge' : 20,
        'Huge' : 25
    },
    11 : {
        'tiny' : 6,
        'scriptsize' : 8,
        'footnotesize' : 9, 
        'small' : 10, 
        'normalsize' : 11,
        'large' : 12, 
        'Large' : 14, 
        'LARGE' : 17,
        'huge' :  20,
        'Huge' :  25
    },
    12 : {
        'tiny' : 6,
        'scriptsize' : 8,
        'footnotesize' : 10, 
        'small' : 11, 
        'normalsize' : 12,
        'large' : 14, 
        'Large' : 17, 
        'LARGE' : 20,
        'huge' :  25,
        'Huge' :  25
    }
}
_width         = 1
_wide_width    = 1
_quantumviolet = '#53257F'
_quantumgray   = '#555555'
def global_setup(columns='twocolumn', paper='a4paper', fontsize=10):
    plt.rcdefaults()
    plt.style.use(['seaborn-v0_8-white', 'quantum-plots.mplstyle']) 
    global _width 
    _width = _widths[columns][paper]
    global _wide_width 
    _wide_width = _wide_widths[columns][paper]
    ## Use the default fontsize scaling of LaTeX
    global _fontsizes
    fontsizes = _fontsizes[fontsize]
    plt.rcParams['axes.labelsize']  = fontsizes['normalsize']
    plt.rcParams['axes.titlesize']  = fontsizes['normalsize']
    plt.rcParams['xtick.labelsize'] = fontsizes['footnotesize']
    plt.rcParams['ytick.labelsize'] = fontsizes['footnotesize']
    plt.rcParams['font.size']       = fontsizes['small']
    plt.rcParams['legend.fontsize'] = fontsizes['footnotesize']
    rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    return {
            'fontsizes' : fontsizes,
            'colors' : {
                'quantumviolet' : _quantumviolet,
                'quantumgray' : _quantumgray
            }
        }

def plot_setup_with_n_subplots(aspect_ratio=1/1.62, width_ratio=1., wide=False, num_subplots=2): 
    """ Horizontal subplots """
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    fig, axs = plt.subplots(1, num_subplots, figsize=(width,height/1.7), dpi=200, facecolor='white')
    return fig, axs

global_setup(columns='twocolumn', paper='a4paper', fontsize=10)

## Define the color and marker settings 
def lighter(color, percent):
    """ Assume the "color" variable is rgb between (0,0,0) and (1,1,1). Function based 
    on https://stackoverflow.com/questions/28015400/how-to-fade-color. """
    color = np.array(color)
    white = colors.to_rgb('white')
    vector = white-color
    return color + vector * percent

blue = 'tab:blue'
light_blue = lighter(colors.to_rgb(blue), 0.7)






################################################################################
#         Plot n(k) vs k for different densities  (one at a time)              #
################################################################################


# N = 80
# densities = [0.069, 0.071]
# rm = 2.9673  # [A]

# path_to_data = f'momentum_distribution_data/N={N}/'
# density_folders = sorted([folder for folder in os.listdir(path_to_data) if any([f'density={density:.3f}' in folder for density in densities])])

# densities = []
# condensate_fractions = []
# condensate_fractions_err = []

# for folder in density_folders:
#     density = float(re.search(r'density=([\d.]+)', folder).group(1))
#     npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]

#     bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)

#     for file in npz_files:
#         data = np.load(os.path.join(path_to_data, folder, file))
#         n_samples = data['n_samples']
#         num_x = data['num_x']
#         n_max = data['n_max']
#         L = data['L']  # [dimensionless]
#         ks = data['ks']/rm  # [A^-1]
#         estimators = data['estimators']

#         ks_norm = jnp.linalg.norm(ks, axis=-1)
#         mask = jnp.argsort(ks_norm)
#         ks_norm_sorted = ks_norm[mask]
#         ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
#         mom_dist_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
#         mom_dist_averaged = jnp.array([jnp.mean(x) for x in mom_dist_splitted])
#         mom_dist_averaged /= jnp.prod(L)  # to compute n_k = N_k/N

#         k = ks_norm_unique
#         nk = mom_dist_averaged
#         bootstrap_samples.append(nk)

#     #     plt.scatter(k, nk)
#     # plt.show()

#     nk = np.mean(np.array(bootstrap_samples), axis=0)  # average over all bootstrap samples (num_ks,)
#     nk_err = np.std(np.array(bootstrap_samples), axis=0) / np.sqrt(len(npz_files))  # standard error (num_ks,)

#     plt.scatter(k, nk)
#     plt.show()







################################################################################
#                Row panel plot of n(k) vs k for many densities                #
################################################################################

# N = 30
# rm = 2.9673  # [A]

# path_to_data = f'momentum_distribution_data/N={N}/'
# density_folders = sorted([folder for folder in os.listdir(path_to_data)])
# # density_folders = sorted([folder for folder in os.listdir(path_to_data) if any([f'density={density:.3f}' in folder for density in densities])])

# fig, axs = plt.subplots(1, len(density_folders), figsize=(25,5))

# for i, folder in enumerate(density_folders):
#     density = float(re.search(r'density=([\d.]+)', folder).group(1))
#     npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]


#     bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)

#     for file in npz_files:
#         data = np.load(os.path.join(path_to_data, folder, file))
#         n_samples = data['n_samples']
#         num_x = data['num_x']
#         n_max = data['n_max']
#         L = data['L']  # [dimensionless]
#         ks = data['ks']/rm  # [A^-1]
#         estimators = data['estimators']

#         ks_norm = jnp.linalg.norm(ks, axis=-1)
#         mask = jnp.argsort(ks_norm)
#         ks_norm_sorted = ks_norm[mask]
#         ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
#         mom_dist_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
#         mom_dist_averaged = jnp.array([jnp.mean(x) for x in mom_dist_splitted])
#         mom_dist_averaged /= jnp.prod(L)  # to compute n_k = N_k/N

#         k = ks_norm_unique
#         nk = mom_dist_averaged
#         bootstrap_samples.append(nk)

#     nk = np.mean(np.array(bootstrap_samples), axis=0)  # average over all bootstrap samples (num_ks,)
#     nk_err = np.std(np.array(bootstrap_samples), axis=0) / np.sqrt(len(npz_files))  # standard error (num_ks,)

#     axs[i].scatter(k, nk)
#     axs[i].set_title(f'$n={density:.3f}$' + '$\ \mathrm{\AA}^{-2}$')

# output_filename = f'panel_plot_momentum-distribution_vs_k_vs_density_N={N}_he4_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=400, bbox_inches='tight')
# plt.tight_layout()
# plt.show()







################################################################################
#          Row panel plot of n(kx,ky) vs (kx,ky) for many densities            #
################################################################################

# N = 30
# rm = 2.9673  # [A]

# path_to_data = f'momentum_distribution_data/N={N}/'
# def extract_density(filename): return float(re.search(r'density=([\d.]+)', filename).group(1))
# density_folders = sorted([folder for folder in os.listdir(path_to_data) if folder.startswith('density')])
# # density_folders = sorted([folder for folder in os.listdir(path_to_data) if extract_density(folder) >= 0.071 and extract_density(folder) < 0.09])

# fig, axs = plt.subplots(1, len(density_folders), figsize=(25,5))

# for i, folder in enumerate(density_folders):
#     print(folder)
#     density = float(re.search(r'density=([\d.]+)', folder).group(1))
#     npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]

#     file = npz_files[0]
#     data = np.load(os.path.join(path_to_data, folder, file))
#     n_samples = data['n_samples']
#     num_x = data['num_x']
#     n_max = data['n_max']
#     L = data['L']  # [dimensionless]
#     ks = data['ks']/rm  # [A^-1]
#     nk = data['estimators'] / jnp.prod(L)  # to compute n_k = N_k/N
#     nk = jnp.real(nk)

#     k_min = jnp.min(ks)
#     k_max = jnp.max(ks)

#     axs[i].imshow(nk.reshape(2*n_max+1, 2*n_max+1), origin='lower', extent=[k_min, k_max, k_min, k_max], cmap='viridis')    
#     axs[i].set_title(f'$n={density:.3f}$' + '$\ \mathrm{\AA}^{-2}$')

# output_filename = f'panel_plot_momentum-distribution_vs_kx-ky_vs_density_N={N}_he4_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=400, bbox_inches='tight')
# plt.tight_layout()
# plt.show()