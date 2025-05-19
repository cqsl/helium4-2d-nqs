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

def plot_setup(aspect_ratio=1/1.62, width_ratio=1., wide=False): 
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    return plt.figure(figsize=(width,height), dpi=200, facecolor='white')

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

markersize = 8
edgewidth = 0.6











# plot_setup()

# def extract_density(filename): return float(re.search(r'density=([\d.]+)', filename).group(1))

# rm = 2.9673  # [A]

# for i, N in enumerate([30,80]):

#     path_to_data = f'data/TO_DELETE_N={N}'

#     densities = []
#     struc_fac_peaks = []

#     for filename in os.listdir(path_to_data):
#         filename_path = os.path.join(path_to_data, filename)
#         data = jnp.load(filename_path)
        
#         density = extract_density(filename)

#         ## Load data
#         n_max = data['n_max']
#         ks = data['ks']/rm  # [A^-1]
#         estimators = data['estimators']
        
#         ## Get the peak at the reciprocal lattice vector, i.e. S(G). This corresponds to
#         ## the 2nd largest value after S(0)
#         S_G = jnp.sort(jnp.real(estimators))[-2]
        
#         densities.append(density)
#         struc_fac_peaks.append(S_G)

#     marker = markers[i]
#     colour = colours[i]
#     light_colour = light_colours[i]
#     plt.scatter(densities, np.array(struc_fac_peaks) / N, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=f'$N={N}$')

# plt.axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
# plt.xlabel('$n \ [\mathrm{\AA}^{-2}]$')
# plt.ylabel('$S(\mathbf{G}) / N$')
# plt.legend()    
# plt.tight_layout()

# output_filename = f'structure_factor_peaks_vs_n_vs_N=30-80_he4_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()











plot_setup()
rm = 2.9673  # [A]
N_axis = [30,80]

for i, N in enumerate(N_axis):
    path_to_data = f'data/N={N}/'

    density_folders = sorted([folder for folder in os.listdir(path_to_data) if folder.startswith('density=')])

    densities = []
    struct_fact_peaks = []
    struct_fact_peaks_err = []

    for folder in density_folders:
        density = float(re.search(r'density=([\d.]+)', folder).group(1))
        
        npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]
        
        bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)
        
        for file in npz_files:
            data = np.load(os.path.join(path_to_data, folder, file))

            ## Load data
            n_max = data['n_max']
            ks = data['ks']/rm  # [A^-1]
            estimators = data['estimators']
            
            ## Get the peak at the reciprocal lattice vector, i.e. S(G). This corresponds to
            ## the 2nd largest value after S(0)
            S_G = jnp.sort(jnp.real(estimators))[-2]
            
            bootstrap_samples.append(S_G)  # add one bootstrap sample

        bootstrapped_S_G = np.mean(bootstrap_samples, axis=0)  # (num_R,)
        bootstrapped_S_G_err = np.std(bootstrap_samples, axis=0) / np.sqrt(len(npz_files)) # (num_R,)

        densities.append(density)
        struct_fact_peaks.append(bootstrapped_S_G)
        struct_fact_peaks_err.append(bootstrapped_S_G_err)

    #     plt.scatter(len(bootstrap_samples)*[density], bootstrap_samples)
    # plt.show()

    marker = markers[i]
    colour = colours[i]
    light_colour = light_colours[i]
    plt.scatter(densities, np.array(struct_fact_peaks) / N ** (5/6), marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=f'$N={N}$')

plt.axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
plt.xlabel('$n \ [\mathrm{\AA}^{-2}]$')
plt.ylabel('$S(\mathbf{G}) / N^{5/6}$')
plt.legend()
# plt.yscale('log')
plt.tight_layout()

output_filename = f'structure_factor_peaks_over_N-5-6_vs_n_vs_N=30-80_he4_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()
