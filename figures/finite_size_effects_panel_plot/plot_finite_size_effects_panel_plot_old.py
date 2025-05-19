import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, colors
import re, os
from scipy.special import erf, ellipe


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
    fig, axs = plt.subplots(1, num_subplots, figsize=(1.7*width,height), dpi=200, facecolor='white')
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

colours = [blue, yellow, green]
light_colours = [light_blue, light_yellow, light_green]

markers = ['o','^','s']

markersize = 8
edgewidth = 0.6






fig, axs = plot_setup_with_n_subplots(num_subplots=2)

rm = 2.9673  # [A]

density = 0.05
filenames = [
    'structure_factor_data/structure_factor_he4_N=16_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1',
    'structure_factor_data/structure_factor_he4_N=30_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1',
    'structure_factor_data/structure_factor_he4_N=80_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1_branch=liquid',
]

for i, filename_head in enumerate(filenames):

    N = int(re.search(r'N=(\d+)', filename_head).group(1))
    density = float(re.search(r'density=([\d.]+)', filename_head).group(1))

    ## Load data
    data = jnp.load(filename_head + '.npz')
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

    # Make a linear fit for small |k|
    num_small_k = 30
    best_fit_params = jnp.polyfit(k[:num_small_k], s[:num_small_k], 1)
    k_fit_axis = jnp.linspace(0, jnp.max(k[:num_small_k]), 100) 
    fit_values = jnp.polyval(p=best_fit_params, x=k_fit_axis)

    ## Plot with an inset
    marker = markers[i]
    colour = colours[i]
    light_colour = light_colours[i]
    axs[0].scatter(k, s/k, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=f'$N={N}$', zorder=len(filenames)-i)

# plt.plot(k_fit_axis, fit_values, '--', color='k', zorder=10)
axs[0].set_xlabel('$k \ [\mathrm{\AA}^{-1}]$')
axs[0].set_ylabel('$S(k)/k \ [\mathrm{\AA}]$')
axs[0].legend()





def energy_size_correction(Lx, Ly, n_max, gamma, density, sk_over_k):
    """ energy correction per particle in"""
    ang = 1e-10  # [m]
    atomic_unit_mass = 1.66053906660 * 10**-27  # [kg]
    m_he4 = 4.00260325413 * atomic_unit_mass  # [kg]
    kb = 1.380649 * 10**-23  # [J/K]
    hbar = 1.054571817 * 10**-34  # [Js]


    alpha = Lx / Ly
    # print(f'alpha = {alpha}')
    integral_term = np.sqrt(np.pi) * ellipe(1-alpha**2) / gamma**(3/2)
    # print(f'integral_term = {integral_term}')

    one_dim_n_axis = np.arange(-n_max, n_max+1, 1) 
    # res = np.sqrt(one_dim_n_axis_squared[None,:] + alpha**2 * one_dim_n_axis_squared[:,None])  # (p,p), p=2*n_max+1
    # res *= np.exp(-gamma * np.linalg.norm()**2)  # (p,p)
    nx, ny = np.meshgrid(one_dim_n_axis, one_dim_n_axis, indexing='ij')
    sum_term = np.sum(np.exp(-gamma * (nx**2 + ny**2)) * np.sqrt(nx**2 + alpha**2 * ny**2))
    # print(f'sum_term = {sum_term}')

    diff = 2 * np.pi * (integral_term - sum_term) / (Lx**2 * Ly) 
    prefactor = hbar**2 / (4 * m_he4 * density * sk_over_k) / ang**2 # [J]
    return prefactor * diff / kb  # [K]



current_dir = os.getcwd()
folderpath = current_dir + '/log/'
alphaHe4 = 1.37643
ns = 512  # number of (optimized) samples to consider 

sk_over_k = {0.05: 0.58, 0.09: 0.1}

## Manually do the plot
res = {
    "N=16":
    {"v_ts": {"n=0.05": -0.8352, "n=0.09": 3.953},
    "v_tsr": {"n=0.05": -0.7302, "n=0.09": 4.552}, 
    "v_p": {"n=0.05": -0.8196, "n=0.09": 4.036}},
    "N=30":
    {"v_ts": {"n=0.05": -0.8028, "n=0.09": 4.212},
    "v_tsr": {"n=0.05": -0.7774, "n=0.09": 4.361},
    "v_p": {"n=0.05": -0.8011, "n=0.09": 4.223}},
    "N=48":
    {"v_ts": {"n=0.05": -0.7963, "n=0.09": None},
    "v_tsr": {"n=0.05": -0.7839, "n=0.09": None},
    "v_p": {"n=0.05": -0.7957, "n=0.09": None}},
    "N=80":
    {"v_ts": {"n=0.05": -0.7925, "n=0.09": 4.305},
    "v_tsr": {"n=0.05": -0.7885, "n=0.09": None},
    "v_p": {"n=0.05": -0.7924, "n=0.09": 4.306}},
}

density = 0.05
N_axis = np.array([16,30,48,80])

potential_types = ['v_p']

deltaE_vs_N = []
N_Mx_My_data = [[16,4,2],[30,5,3],[48,6,4],[80,8,5]]  # (N,Mx,My) 
for N, Mx, My in N_Mx_My_data:
    assert N == 2*Mx*My
    a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
    # lx x ly defines a rectangle that contains two unit cells (in terms of its area)
    lx = a  # [A]
    ly = np.sqrt(3)*a  # [A]
    Lx = Mx*lx  # [A]
    Ly = My*ly  # [A]
    print(f'N={N}, Lx={Lx:.2f} A, Ly={Ly:.2f} A')
    dE = energy_size_correction(Lx, Ly, n_max=500, gamma=0.001, density=density, sk_over_k=sk_over_k[density])
    deltaE_vs_N.append(dE)
deltaE_vs_N = np.array(deltaE_vs_N)
print(f'deltaE = {deltaE_vs_N} K')

for i, potential_type in enumerate(potential_types):
    energies_vs_N = np.array([res[f"N={N}"][potential_type][f"n={density:.2f}"] for N in N_axis])
    axs[1].scatter(1/N_axis**(3/2), energies_vs_N, marker=markers[0], color=light_colours[0], edgecolors=colours[0], linewidths=edgewidth, s=markersize, label='$\\varepsilon_N$')
    axs[1].scatter(1/N_axis**(3/2), energies_vs_N + deltaE_vs_N, marker=markers[2], color=light_colours[2], edgecolors=colours[2], linewidths=edgewidth, s=markersize, label='$\\varepsilon_N + \delta \\varepsilon_N$')
axs[1].set_xlabel('$1/N^{3/2}$')
axs[1].set_ylabel('$E/N$ [K]')
axs[1].legend()
plt.xlim(0,0.017)

plt.tight_layout()

output_filename = 'finite_size_effect_panel_plot_structure_factor_and_energy_vs_1-N_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()
