import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import re
import os



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

# Sets up the plot with the fitting arguments so that the font sizes of the plot
# and the font sizes of the document are well aligned
#
#     aspect_ratio : float
#         the aspect ratio (width/height) of your plot
#         defaults to the golden ratio
#
#     width_ratio : float in [0, 1]
#         the width of your plot when you insert it into the document, e.g.
#         .8 of the regular width
#         defaults to 1.0
#
#     wide : bool 
#         indicates if the figures spans two columns in twocolumn mode, i.e.
#         when the figure* environment is used, has no effect in onecolumn mode 
#         defaults to False
#
#     (returns) : matplotlib figure object
#         the initialized figure object
def plot_setup(aspect_ratio=1/1.62, width_ratio=1., wide=False): 
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    return plt.figure(figsize=(width,height), dpi=200, facecolor='white')

def plot_setup_with_n_subplots(aspect_ratio=1/1.62, width_ratio=1., wide=False, num_subplots=2): 
    """ Horizontal subplots """
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    #### MIGHT NEED TO TUNE THE `figsize` ARGUMENT ####
    fig, axs = plt.subplots(1, num_subplots, figsize=(20,3), dpi=200, facecolor='white')
    return fig, axs

global_setup(columns='twocolumn', paper='a4paper', fontsize=10)




N = 30
rm = 2.9673  # [A]
path = f'g6_data/N={N}/'




############################################################
#               Plot g6(r) vs r (vs density)               #
############################################################



fig = plot_setup()
ax = plt.gca()
ax.set_yscale('log')

def extract_density(filename):
    return float(re.search(r'density=(\d+\.\d+)', filename).group(1))
filenames = os.listdir(path)
sorted_filenames = sorted(filenames, key=extract_density)

num_dataset = len(sorted_filenames)
color_axis = np.linspace(0,1,num_dataset)
colors = plt.cm.viridis(color_axis)

densities = []
for i, filename_head in enumerate(sorted_filenames):
    if filename_head.startswith('hexatic') and f'N={N}' in filename_head and filename_head.endswith('.npz'): 

        N = int(re.search(r'N=(\d+)', filename_head).group(1))
        density = float(re.search(r'density=([\d.]+)', filename_head).group(1))

        ## Load data
        filename_path = os.path.join(path, filename_head)
        data = jnp.load(filename_path)
        # n_samples = data['n_samples']
        num_r = data['num_r']
        radius_axis = data['radius_axis']
        estimators = data['estimators']
        densities.append(density)

        estimators = np.real(estimators)    
        plt.plot(rm * radius_axis, estimators, color=colors[i], marker='o', markersize=1.5,)# label=f'$N$={N}, $n$={density} '+'$\AA^{-2}$')


plt.xlabel('$r \ [\mathrm{\AA}]$')
plt.ylabel('$g_6(r)$')
# plt.legend()

plt.ylim(ymin=1e-4, ymax=1e1)

# cbar = plt.colorbar(plt.scatter(densities, densities, c=densities, cmap='viridis'))
norm = Normalize(vmin=min(densities), vmax=max(densities))
mappable = ScalarMappable(norm=norm, cmap='viridis')
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label('$n \ [\AA^{-2}]$')

output_filename = f'g6_vs_r_vs_density_he4_N={N}_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
# plt.savefig(output_filename, dpi=400, bbox_inches='tight')
plt.show()
















# ############################################################
# #                 Check decay of the peaks                 #
# ############################################################


def extract_density(filename):
    return float(re.search(r'density=(\d+\.\d+)', filename).group(1))
filenames = os.listdir(path)
filenames = [f for f in filenames if f.startswith('hexatic') and f'N={N}' in f and f.endswith('.npz')]
# filenames = [f for f in filenames if 0.065 <= extract_density(f) <= 0.075]
sorted_filenames = sorted(filenames, key=extract_density)

# fig, axs = plot_setup_with_n_subplots(num_subplots=len(sorted_filenames))
fig, axs = plt.subplots(1, len(sorted_filenames), figsize=(15,3))

densities = []
for i, filename_head in enumerate(sorted_filenames):

    # N = int(re.search(r'N=(\d+)', filename_head).group(1))
    density = float(re.search(r'density=([\d.]+)', filename_head).group(1))

    ## Load data
    filename_path = os.path.join(path, filename_head)
    data = jnp.load(filename_path)
    # n_samples = data['n_samples']
    num_r = data['num_r']
    radius_axis = data['radius_axis']
    estimators = data['estimators']
    densities.append(density)

    estimators = np.real(estimators)
    axs[i].plot(rm * radius_axis, estimators)



    def find_local_minima_maxima(data):
        derivative = np.diff(data)
        sign_changes = np.sign(derivative[:-1]) != np.sign(derivative[1:])
        minima_indices= np.where((sign_changes) & (derivative[:-1] < 0))[0] + 1
        maxima_indices = np.where((sign_changes) & (derivative[:-1] > 0))[0] + 1
        return minima_indices, maxima_indices

    _, maxima_indices = find_local_minima_maxima(estimators)

    # from scipy.signal import find_peaks
    # maxima_indices, _ = find_peaks(estimators, prominence=0.01)

    ## Filter the local maxima based on a range of interest
    if N == 30:
        r_min = 3./rm
        r_max = 10./rm
    elif N == 80:
        r_min = 3./rm
        r_max = 9.2/rm

    ## Focus on the peaks in a given radius range
    maxima_indices = [i for i in maxima_indices if r_min <= radius_axis[i] <= r_max]

    ## Sort maxima_indices by their values in estimators
    sorted_maxima_indices = sorted(maxima_indices, key=lambda i: estimators[i], reverse=True)

    ## Eliminate local maximum indices that are in the same neighborhood
    neighborhood = 10
    filtered_maxima_indices = []
    for k in sorted_maxima_indices:
        if not filtered_maxima_indices or abs(k - filtered_maxima_indices[-1]) >= neighborhood:
            filtered_maxima_indices.append(k)

    ## Select the 2nd largest peak
    peak_index = filtered_maxima_indices[1] 
    peak_x = rm * radius_axis[peak_index]
    peak_y = estimators[peak_index]

    # Plot the power law line 1/r^{1/4} starting from the first local maximum
    r_values = np.linspace(1, max(rm * radius_axis), 100)
    power_law_values = peak_y * (peak_x / r_values)**(1/4)
    axs[i].plot(r_values, power_law_values, color='red', linestyle='--', label='$\sim 1/r^{1/4}$')

    # Add a red star on the local maximum identified
    axs[i].scatter(peak_x, peak_y, color='red', marker='*', s=10, zorder=10) # , label='First Local Maximum'
    # axs[i].scatter(rm * radius_axis[maxima_indices], estimators[maxima_indices], color='blue', marker='*', s=10, zorder=10) # , label='First Local Maximum'

    axs[i].set_xlabel('$r \ [\mathrm{\AA}]$')
    axs[i].set_title(f'$n={density}$' + '$\ \mathrm{\AA}^{-2}$')
    axs[i].set_ylim(ymin=-0.2, ymax=0.45)


axs[0].set_ylabel('$g_6(r)$')

output_filename = f'g6_vs_r_with_1-r-1-4_vs_density_he4_N={N}_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
# plt.savefig(output_filename, dpi=400, bbox_inches='tight')
plt.show()

