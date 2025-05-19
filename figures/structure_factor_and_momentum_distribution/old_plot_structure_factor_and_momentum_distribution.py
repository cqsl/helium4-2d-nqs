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
green = convert_rgb_255_to_1_scale((0, 153, 68))

light_blue = lighter(colors.to_rgb(blue), 0.7)
light_green = lighter(colors.to_rgb(green), 0.7)

colours = [blue, green]
light_colours = [light_blue, light_green]

markers = ['o','s']

markersize = 7
edgewidth = 0.3




##################################################################
#             Inset of the condensate fraction plot              #
##################################################################

rm = 2.9673  # [A]
path = f'data/'

def extract_density(filename): return float(re.search(r'density=([\d.]+)', filename).group(1))
def extract_N(filename): return int(re.search(r'N=(\d+)', filename).group(1))

filenames = os.listdir(path)
density1 = 0.069
density2 = 0.071
filenames = [f for f in filenames if f.endswith('.npz') and f'density={density1:.3f}' in f or f'density={density2:.3f}' in f]
fig, axs = plot_setup_with_2x2_subplots()

densities = []
for filename in filenames:

    density = extract_density(filename)
    N = extract_N(filename)
    phase = 'liquid' if 'liquid' in filename else 'solid'

    if density not in densities:
        densities.append(density)

    colour = colours[0] if N == 30 else colours[1]
    light_colour = light_colours[0] if N == 30 else light_colours[1]
    marker = markers[0] if N == 30 else markers[1]
    zorder = 1 if N == 30 else 0

    filename_path = os.path.join(path, filename)
    data = jnp.load(filename_path)

    if 'structure_factor' in filename and N == 80:

        ax = axs[0,0] if density == density1 else axs[0,1] 

        ## Load data
        n_max = data['n_max']
        ks = data['ks']/rm  # [A^-1]
        estimators = data['estimators']

        # ## S(|k|) vs |k|
        # ks_norm = jnp.linalg.norm(ks, axis=-1)
        # mask = jnp.argsort(ks_norm)
        # ks_norm_sorted = ks_norm[mask]
        # ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
        # struc_fac_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
        # struc_fac_averaged = jnp.array([jnp.mean(x) for x in struc_fac_splitted])

        # # Avoid |k|=0
        # k = ks_norm_unique[1:]
        # s = struc_fac_averaged[1:]

        # ax.scatter(k, s, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, zorder=zorder)

        # # Remove the x-labels (it will be put on the momentum distrubution plots)
        # ax.set_xticklabels([])


        ks_norm = jnp.linalg.norm(ks, axis=-1)
        k_min = jnp.min(ks)  # [A^-1]
        k_max = jnp.max(ks)  # [A^-1]
        print(f'k_min={k_min}, k_max={k_max}')
        estimators = jnp.real(estimators)
        estimators = estimators.at[jnp.argmax(estimators)].set(0.)  # smooth out the gamma point k=(0,0)
        im = ax.imshow(estimators.reshape(2*n_max+1, 2*n_max+1), extent=[k_min,k_max,k_min,k_max], origin='lower', cmap='viridis', vmin=0,)# vmax=max_sk)
        ax.set_xlabel('$k_x \ [\mathrm{\AA}^{-1}]$',)# fontsize=8)
        ax.set_ylabel('$k_y \ [\mathrm{\AA}^{-1}]$',)# fontsize=8)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('$S(k_x,k_y)$',)# fontsize=8)


    elif 'momentum_distribution' in filename:

        ax = axs[1,0] if density == density1 else axs[1,1]  

        ## Load data
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
        # sum_Nk = jnp.sum(Nk)
        # scaling_factor = N/sum_Nk
        # Nk *= scaling_factor
        # nk = Nk/N  # normalize n(k) as well
        # # print(f'The following should be equal to {N}: {jnp.sum(Nk)}')

        ax.scatter(k[0:], Nk[0:], facecolors=light_colour, edgecolors=colour, linewidths=edgewidth, marker=marker, s=markersize, zorder=zorder, label=f'$n$={density} '+'$\AA^{-2}$')  # for insets:  s=150, color='tab:orange', 

        ax.set_xlabel('$k \ [\mathrm{\AA}^{-1}]$',)# fontsize=8)
        ax.set_ylabel('$N_k$',)# fontsize=8)   

        ax.set_xlim(-0.2,4.)  
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))  # specify the number of y-ticks 
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))  
        ax.tick_params(axis='x',)# labelsize=8)
        ax.tick_params(axis='y',)# labelsize=8)


output_filename = f'structure_factor_and_momentum_distribution_vs_k_vs_N=30-80_he4_d=2_density={density1:.3f}_{density2:.3f}_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
plt.subplots_adjust(wspace=0.7)  # Adjust this value to increase/decrease horizontal spacing
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
# density1 = 0.067
# filenames = [f for f in filenames if f.endswith('.npz') and f'density={density1:.3f}' in f and f'N={N}' in f]
# fig, axs = plot_setup_with_1x2_subplots()


# for filename in filenames:

#     density = extract_density(filename)
#     N = extract_N(filename)
#     phase = 'liquid' if 'liquid' in filename else 'solid'

#     colour = colours[0] if N == 30 else colours[1]
#     light_colour = light_colours[0] if N == 30 else light_colours[1]
#     marker = markers[0] if N == 30 else markers[1]
#     zorder = 1 if N == 30 else 0

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
# print(f'volume={jnp.prod(L)/rm**2}')
# print(f'The following should be equal to {N}: {jnp.sum(jnp.real(Nk))}')
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
