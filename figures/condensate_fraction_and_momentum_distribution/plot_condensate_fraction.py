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

def plot_with_inset_setup(aspect_ratio=1/1.62, width_ratio=1., wide=False):
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width,height), dpi=200, facecolor='white')
    # these are in unitless percentages of the figure size. (0,0 is bottom left):
    left, bottom, width, height = [0.57, 0.57, 0.38, 0.35] 
    ax_inset = fig.add_axes([left, bottom, width, height])
    return ax, ax_inset

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

colours = [blue, yellow, green]
light_colours = [light_blue, light_yellow, light_green]

markers = ['o','^','s']

markersize = 8
edgewidth = 0.6
markeredgewidth = 0.5
errorbarwidth = markeredgewidth
capsize = 2







# rm = 2.9673  # [A]
# path = 'TO_DELETE_momentum_distribution_data/'

# # Get a list of all files in the s directory
# files = os.listdir(path)

# def extract_density(filename):
#     return float(re.search(r'density=(\d+\.\d+)', filename).group(1))
# filenames = os.listdir(path)
# sorted_filenames = sorted([f for f in filenames if f.endswith('.npz') and f.startswith('momentum')], key=extract_density)


# densities = {"N=30": [], "N=80": []}
# condensate_fractions = {"N=30": [], "N=80": []}

# ## For n(k) vs k plot
# fig = plot_setup()

# for i, filename_head in enumerate(sorted_filenames):

#     N = int(re.search(r'N=(\d+)', filename_head).group(1))
#     density = float(re.search(r'density=(\d+\.\d+)', filename_head).group(1))  # [A^-d]

#     ## Load data
#     filename_path = os.path.join(path, filename_head)
#     data = jnp.load(filename_path)
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
#     condensate_fraction = nk[0]

#     # Nk = N * nk
#     # sum_Nk = jnp.sum(Nk)
#     # scaling_factor = N/sum_Nk
#     # Nk *= scaling_factor
#     # nk = Nk/N  # normalize n(k) as well
#     # # print(f'The following should be equal to {N}: {jnp.sum(Nk)}')
#     # condensate_fraction = Nk[0]/N
#     # # print(f'N0 = {condensate_fraction}')
#     # print(condensate_fraction)

#     densities[f"N={N}"].append(density)
#     condensate_fractions[f"N={N}"].append(condensate_fraction)


# ## Plot condensate fraction vs density
# for N in densities.keys():
#     color = blue if N == "N=30" else orange
#     light_color = light_blue if N == "N=30" else light_orange
#     marker = markers[0] if N == "N=30" else markers[1]
#     plt.scatter(densities[N], condensate_fractions[N], marker=marker, facecolors=light_color, edgecolors=color, linewidths=0.6, label=f'${N}$')
#     plt.yscale('log')

# plt.axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')

# plt.xlabel('$n \ [\mathrm{\AA}^{-2}]$')
# plt.ylabel('Condensate fraction')
# # plt.ylim(0,0.4)

# output_filename = 'condensate_fraction_vs_density_N=30-80_he4_d=2_bt=4_ld=8_nf=8_hd=1'
# plt.tight_layout()
# plt.legend()
# # plt.savefig(output_filename + '.pdf', dpi=300, bbox_inches='tight')
# plt.show()










#######################################################################################################
#       Plot condensate fraction vs number density with error bars (obtained by bootstrapping)        #  
#######################################################################################################


N = 80
rm = 2.9673  # [A]
fig = plot_setup()

N_axis = [30,56,80]

for i, N in enumerate(N_axis):
    path_to_data = f'momentum_distribution_data/N={N}/'
    density_folders = sorted([folder for folder in os.listdir(path_to_data) if folder.startswith('density=')])

    densities = []
    condensate_fractions = []
    condensate_fractions_err = []

    for folder in density_folders:
        density = float(re.search(r'density=([\d.]+)', folder).group(1))

        npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]
        
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
            condensate_fraction = nk[0]
            bootstrap_samples.append(condensate_fraction)  # add one bootstrap sample

        bootstrapped_n0 = np.mean(bootstrap_samples, axis=0)  # (num_R,)
        bootstrapped_n0_err = np.std(bootstrap_samples, axis=0) / np.sqrt(len(npz_files)) # (num_R,)

        densities.append(density)
        condensate_fractions.append(bootstrapped_n0)
        condensate_fractions_err.append(bootstrapped_n0_err)

    #     plt.scatter(len(bootstrap_samples)*[density], bootstrap_samples)
    # plt.show()

    marker = markers[i]
    colour = colours[i]
    light_colour = light_colours[i]
    # plt.errorbar(
    #     densities,
    #     condensate_fractions,
    #     yerr=condensate_fractions_err,
    #     color=light_colour,
    #     ecolor=colour,
    #     elinewidth=errorbarwidth,
    #     capsize=capsize,
    #     marker=marker,
    #     markeredgecolor=colour,
    #     markeredgewidth=markeredgewidth,
    #     markersize=markersize,
    #     linestyle='None',
    #     label=f'$N={N}$',
    # )
    plt.scatter(densities, condensate_fractions, marker=marker, color=light_colour, edgecolors=colour, linewidths=markeredgewidth, s=markersize, label=f'$N={N}$')

plt.axvspan(0.06811, 0.07159, alpha=0.2, color='gray',)# label='Obtained with the N=80 data')
# plt.axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
plt.xlabel('$n \ [\mathrm{\AA}^{-2}]$')
plt.ylabel('Condensate fraction')
plt.tight_layout()
plt.legend()
plt.yscale('log')

output_filename = 'condensate_fraction_vs_density_N=30-56-80_he4_d=2_bt=4_ld=8_nf=8_hd=1'
# plt.savefig(output_filename + '.pdf', dpi=300, bbox_inches='tight')
plt.show()
