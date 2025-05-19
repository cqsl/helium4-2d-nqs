import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, colors
import re, os



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

def plot_setup_with_2x2_subplots(aspect_ratio=1/1.62, width_ratio=1., wide=False): 
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    fig, axs = plt.subplots(2, 2, figsize=(1.3*width,1.3*height), dpi=200, facecolor='white')
    return fig, axs

def plot_setup_with_1x2_subplots(aspect_ratio=1/1.62, width_ratio=1., wide=False): 
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    fig, axs = plt.subplots(1, 2, figsize=(1.3*width,height), dpi=200, facecolor='white')
    return fig, axs

def plot_setup_with_2x3_subplots(aspect_ratio=1/1.62, width_ratio=1., wide=False): 
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    fig, axs = plt.subplots(2, 3, figsize=(1.7*width,1.5*height), dpi=200, facecolor='white')
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

def convert_rgb_255_to_1_scale(color: tuple):
    return tuple(np.array(color)/255)

blue = convert_rgb_255_to_1_scale((0,119,187))
green = convert_rgb_255_to_1_scale((0,153,68))
yellow = convert_rgb_255_to_1_scale((255,187,0))

light_blue = lighter(colors.to_rgb(blue), 0.7)
light_green = lighter(colors.to_rgb(green), 0.7)
light_yellow = lighter(colors.to_rgb(yellow), 0.7)

colours = {'N=30': blue, 'N=56': yellow, 'N=80': green}
light_colours = {'N=30': light_blue, 'N=56': light_yellow, 'N=80': light_green} 
markers = {'N=30': 'o', 'N=56': '^', 'N=80': 's'}
zorders = {'N=30': 3, 'N=56': 2, 'N=80': 1}  # since smaller N's have less data points, make them appear on top

markersize = 7
edgewidth = 0.3














##################################################################
#             Inset of the condensate fraction plot              #
##################################################################


rm = 2.9673  # [A]

def extract_density(filename): return float(re.search(r'density=([\d.]+)', filename).group(1))
def extract_N(filename): return int(re.search(r'N=(\d+)', filename).group(1))

fig, axs = plot_setup_with_2x3_subplots()

## First row of the inset contains 3 subplots with S(kx,ky) vs kx,ky:
## (1) N=30, density=0.067;
## (2) N=80, density=0.069;
## (3) N=80, density=0.071;

first_row_vars = [[0,30,0.067],[1,80,0.069],[2,80,0.071]]  # of the form (i,N,density)

for var in first_row_vars:

    i, N, density = var
    ax = axs[0,i]

    path_to_data = f'data/structure_factor_data/N={N}/'
    folder = sorted([folder for folder in os.listdir(path_to_data) if f'density={density:.3f}' in folder])[0]

    npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]
    filename = npz_files[0]

    data = np.load(os.path.join(path_to_data, folder, filename))

    ## Load data
    n_max = data['n_max']
    ks = data['ks']/rm  # [A^-1]
    estimators = data['estimators']

    ks_norm = jnp.linalg.norm(ks, axis=-1)
    k_min = jnp.min(ks)  # [A^-1]
    k_max = jnp.max(ks)  # [A^-1]

    estimators = jnp.real(estimators)
    estimators = estimators.at[jnp.argmax(estimators)].set(0.)  # smooth out the gamma point k=(0,0)
    im = ax.imshow(estimators.reshape(2*n_max+1, 2*n_max+1), extent=[k_min,k_max,k_min,k_max], origin='lower', cmap='viridis', vmin=0,)# vmax=max_sk)
    ax.set_xlabel('$k_x \ [\mathrm{\AA}^{-1}]$', labelpad=0.)# fontsize=8)
    if i == 0: 
        ax.set_ylabel('$k_y \ [\mathrm{\AA}^{-1}]$', labelpad=0.)# fontsize=8)

    ax.text(0.5, 0.97, f'$N={N}$', transform=ax.transAxes, color='white', verticalalignment='top', horizontalalignment='center', fontsize=11)
    ax.set_title(f'$n={density:.3f}$' + '$ \ \mathrm{\AA}^{-2}$')

    cbar = fig.colorbar(im, ax=ax)
    if i == 2:
        cbar.set_label('$S(k_x,k_y)$',)# fontsize=8)



## Second row of the inset contains 3 subplots of N_k vs k:
## (1) N=30,80, density=0.067;
## (1) N=30,56,80, density=0.069;
## (1) N=30,56,80, density=0.071;

N_axis = [30,56,80]
densities = [0.067,0.069,0.071]

for i, density in enumerate(densities):
    
    ax = axs[1,i]

    for N in N_axis:

        path_to_data = f'data/momentum_distribution_data/N={N}/'
        folder = sorted([folder for folder in os.listdir(path_to_data) if f'density={density:.3f}' in folder])[0]

        density = float(re.search(r'density=([\d.]+)', folder).group(1))
        npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]

        colour = colours[f'N={N}']
        light_colour = light_colours[f'N={N}']
        marker = markers[f'N={N}']
        zorder = zorders[f'N={N}']

        bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)

        for file in npz_files:
            data = np.load(os.path.join(path_to_data, folder, file))
            n_samples = data['n_samples']
            num_x = data['num_x']
            n_max = data['n_max']
            L = data['L']  # [dimensionless]
            ks = data['ks']/rm  # [A^-1]
            estimators = data['estimators']

            ks_norm = jnp.linalg.norm(ks, axis=-1)
            mask = jnp.argsort(ks_norm)
            ks_norm_sorted = ks_norm[mask]
            ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
            mom_dist_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
            mom_dist_averaged = jnp.array([jnp.mean(x) for x in mom_dist_splitted])
            mom_dist_averaged /= jnp.prod(L)  # to compute n_k = N_k/N

            k = ks_norm_unique
            nk = mom_dist_averaged
            Nk = N * nk
            bootstrap_samples.append(Nk)


        Nk = np.mean(np.array(bootstrap_samples), axis=0)  # average over all bootstrap samples (num_ks,)
        Nk_err = np.std(np.array(bootstrap_samples), axis=0) / np.sqrt(len(npz_files))  # standard error (num_ks,)
        
        ax.scatter(k[0:], Nk[0:], facecolors=light_colour, edgecolors=colour, linewidths=edgewidth, marker=marker, s=markersize, zorder=zorder) 

        
        ax.set_xlabel('$k \ [\mathrm{\AA}^{-1}]$', labelpad=0.)# fontsize=8)
        if i == 0:
            ax.set_ylabel('$N_k$',)# fontsize=8)   

    # ax.set_xlim(-0.2,4.)  
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))  # specify the number of y-ticks 
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))  
    # ax.tick_params(axis='x',)# labelsize=8)
    # ax.tick_params(axis='y',)# labelsize=8)




density_str = '_'.join([f'{density:.3f}' for density in densities])
output_filename = f'structure_factor_and_momentum_distribution_vs_k_vs_N=30-80_he4_d=2_density={density_str}_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # Adjust this value to increase/decrease horizontal spacing
# plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()













##################################################################
#                Superhexatic evidence for N=30                  #
##################################################################

# N = 30
# rm = 2.9673  # [A]
# path = f'data/'

# def extract_density(filename): return float(re.search(r'density=([\d.]+)', filename).group(1))
# def extract_N(filename): return int(re.search(r'N=(\d+)', filename).group(1))

# filenames = os.listdir(path)
# density1 = 0.069
# filenames = [f for f in filenames if f.endswith('.npz') and f'density={density1:.3f}' in f and f'N={N}' in f]
# fig, axs = plot_setup_with_1x2_subplots()


# for filename in filenames:

#     density = extract_density(filename)
#     N = extract_N(filename)
#     phase = 'liquid' if 'liquid' in filename else 'solid'

#     colour = colours[f'N={N}']
#     light_colour = light_colours[f'N={N}']
#     marker = markers[f'N={N}']
#     zorder = zorders[f'N={N}']

#     filename_path = os.path.join(path, filename)
#     data = jnp.load(filename_path)

#     if 'structure_factor' in filename:

#         ax = axs[0]

#         ## Load data
#         n_max = data['n_max']
#         ks = data['ks']/rm  # [A^-1]
#         estimators = data['estimators']

#         ks_norm = jnp.linalg.norm(ks, axis=-1)
#         k_min = jnp.min(ks)  # [A^-1]
#         k_max = jnp.max(ks)  # [A^-1]
#         estimators = jnp.real(estimators)
#         estimators = estimators.at[jnp.argmax(estimators)].set(0.)  # smooth out the gamma point k=(0,0)
#         im = ax.imshow(estimators.reshape(2*n_max+1, 2*n_max+1), extent=[k_min,k_max,k_min,k_max], origin='lower', cmap='viridis', vmin=0,)# vmax=max_sk)
#         ax.set_xlabel('$k_x \ [\mathrm{\AA}^{-1}]$',)# fontsize=8)
#         ax.set_ylabel('$k_y \ [\mathrm{\AA}^{-1}]$',)# fontsize=8)

#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label('$S(k_x,k_y)$',)# fontsize=8)



# path_to_data = f'data/momentum_distribution_data/N={N}/'
# density_folders = sorted([folder for folder in os.listdir(path_to_data)])
# folder = sorted([folder for folder in os.listdir(path_to_data) if f'density={density1:.3f}' in folder])[0]

# density = float(re.search(r'density=([\d.]+)', folder).group(1))
# npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]

# bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)

# for file in npz_files:
#     data = np.load(os.path.join(path_to_data, folder, file))
#     n_samples = data['n_samples']
#     num_x = data['num_x']
#     n_max = data['n_max']
#     L = data['L']  # [dimensionless]
#     ks = data['ks']/rm  # [A^-1]
#     estimators = data['estimators']

#     ks_norm = jnp.linalg.norm(ks, axis=-1)
#     mask = jnp.argsort(ks_norm)
#     ks_norm_sorted = ks_norm[mask]
#     ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
#     mom_dist_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
#     mom_dist_averaged = jnp.array([jnp.mean(x) for x in mom_dist_splitted])
#     mom_dist_averaged /= jnp.prod(L)  # to compute n_k = N_k/N

#     k = ks_norm_unique
#     nk = mom_dist_averaged
#     bootstrap_samples.append(nk)

# nk = np.mean(np.array(bootstrap_samples), axis=0)  # average over all bootstrap samples (num_ks,)
# nk_err = np.std(np.array(bootstrap_samples), axis=0) / np.sqrt(len(npz_files))  # standard error (num_ks,)

# Nk = N * nk

# dkx = 0.3071848
# dky = 0.29558871
# dVk = dkx * dky  # [A^-2]
# volume = jnp.prod(L) * rm**2  # [A^2]

# res = (volume * dVk / (2*jnp.pi)**2) * np.sum(Nk)  # should be equal to N
# print((volume * dVk / (2*jnp.pi)**2))

# print(f'The following should be equal to {N}: {jnp.sum(Nk)}')
# axs[1].scatter(k[0:], Nk[0:], facecolors=light_colour, edgecolors=colour, linewidths=edgewidth, marker=marker, s=markersize)  

# axs[1].set_xlabel('$k \ [\mathrm{\AA}^{-1}]$',)# fontsize=8)
# axs[1].set_ylabel('$N_k$',)# fontsize=8) 



# output_filename = f'structure_factor_vs_kx-ky_and_momentum_distribution_vs_k_vs_N=30-80_he4_d=2_density={density1:.3f}_bt=4_ld=8_nf=8_hd=1.pdf'
# plt.tight_layout()
# # plt.subplots_adjust(wspace=0.7)  # Adjust this value to increase/decrease horizontal spacing
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()















##############################################################################################
#     The code below works only if there are 4 files in the `data` folder for a fixed N      #
##############################################################################################

# rm = 2.9673  # [A]
# N = 30
# path = f'data/'
# def extract_density(filename):
#     return float(re.search(r'density=([\d.]+)', filename).group(1))
# filenames = os.listdir(path)
# filenames = [filename for filename in filenames if filename.endswith('.npz')]
# def sort_key(filename):
#     if 'structure_factor' in filename:
#         return (0, extract_density(filename))
#     elif 'momentum_distribution' in filename:
#         return (1, extract_density(filename))
#     else:
#         return (2, extract_density(filename))
# sorted_filenames = sorted(filenames, key=sort_key)

# fig, axs = plot_setup_with_2x2_subplots()
# densities = []

# for i, filename in enumerate(sorted_filenames):

#     phase = 'liquid' if 'liquid' in filename else 'solid'
#     # N = int(re.search(r'N=(\d+)', filename).group(1))
#     density = extract_density(filename)
#     if density not in densities:
#         densities.append(density)

#     filename_path = os.path.join(path, filename)
#     data = jnp.load(filename_path)

#     ax = axs[i//2, i%2] 

#     if 'structure_factor' in filename:

#         ## Load data
#         n_max = data['n_max']
#         ks = data['ks']/rm  # [A^-1]
#         estimators = data['estimators']

#         ## S(|k|) vs |k|
#         ks_norm = jnp.linalg.norm(ks, axis=-1)
#         mask = jnp.argsort(ks_norm)
#         ks_norm_sorted = ks_norm[mask]
#         ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
#         struc_fac_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
#         struc_fac_averaged = jnp.array([jnp.mean(x) for x in struc_fac_splitted])

#         # Avoid |k|=0
#         k = ks_norm_unique[1:]
#         s = struc_fac_averaged[1:]

#         ax.scatter(k, s, color=light_blue, edgecolors=blue, linewidths=edgewidth, s=markersize)

#         ## Only put a y-label on the leftmost plot, and remove the x-labels (it will be put on the momentum distrubution plots)
#         if i == 0:
#             ax.set_ylabel('$S(k)$', fontsize=8)
#         ax.set_xticklabels([])

#     elif 'momentum_distribution' in filename:

#         ## Load data
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

#         Nk = N * nk
#         sum_Nk = jnp.sum(Nk)
#         scaling_factor = N/sum_Nk
#         Nk *= scaling_factor
#         nk = Nk/N  # normalize n(k) as well
#         # print(f'The following should be equal to {N}: {jnp.sum(Nk)}')

#         ax.scatter(k[0:], nk[0:], facecolors=light_blue, edgecolors=blue, linewidths=edgewidth, s=markersize, label=f'$n$={density} '+'$\AA^{-2}$')  # for insets:  s=150, color='tab:orange', 

#         if i == len(sorted_filenames)//2:
#             ax.set_ylabel('$n(k)$', fontsize=8)
#         ax.set_xlabel('$k \ [\AA^{-1}]$', fontsize=8)

#     ax.set_xlim(-0.2,4.)  # set the same x-axis limit to be the same for the structure factor and the momentum distribution plots
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))  # specify the number of y-ticks 
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))  
#     ax.tick_params(axis='x', labelsize=8)
#     ax.tick_params(axis='y', labelsize=8)

# output_filename = f'structure_factor_and_momentum_distribution_vs_k_N=30_he4_d=2_density={densities[0]:.3f}_{densities[1]:.3f}_bt=4_ld=8_nf=8_hd=1.pdf'
# plt.tight_layout()
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()
