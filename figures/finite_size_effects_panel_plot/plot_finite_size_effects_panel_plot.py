import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, colors
import re, os, json
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
red = convert_rgb_255_to_1_scale((186, 30, 15))

light_blue = lighter(colors.to_rgb(blue), 0.7)
light_green = lighter(colors.to_rgb(green), 0.7)
light_yellow = lighter(colors.to_rgb(yellow), 0.7)

colours = [blue, yellow, green]
light_colours = [light_blue, light_yellow, light_green]

markers = ['o','^','s']

markersize = 8
markersize_right = 14
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
    axs[0].scatter(k, s, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=f'$N={N}$', zorder=len(filenames)-i)

# plt.plot(k_fit_axis, fit_values, '--', color='k', zorder=10)
axs[0].set_xlabel('$k \ [\mathrm{\AA}^{-1}]$')
axs[0].set_ylabel('$S(k)$')
axs[0].legend()










###########################################################################################
#                      Plot E/N vs N^{-3/2} (for a fixed density)                         #
###########################################################################################


rm = 2.9673  # [A]

Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}

N_axis = [30,56,80]
res = {}  # to store all the data in there
for N in N_axis: res[f'N={N}'] = {}

def extract_density(filename): return float(re.search(r'density=([\d.]+)', filename).group(1))


working_dir = os.getcwd()
alpha = 1.37643  # hbar^2/(m*r_m^2*k_B) [K]
ns = 1024  # number of values to consider for the average
potential_type_subfolder = 'shifted_truncated_pot' # 'full_periodized_pot' 


n = 0.07  # fixed a number density [A^-2]
n_axis = [0.065,0.066,0.067,0.068,0.069,0.07,0.071,0.072,0.073,0.075]
E_infs = []
for n in n_axis:

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
    print(f'E_inf/N = {E_inf:.5f} K')
    E_infs.append(E_inf)

    delta_E = abs(E_inf - y)
    print(f'N = {N_axis}, \ndelta_E/N = {delta_E} K')   

    # Generate line values
    x_fit = np.linspace(0, max(x), 100)
    y_fit = slope * x_fit + intercept

    # Plot the energy and the extrapolated energy
    # for i in range(len(N_axis)):
    #     marker = markers[i]
    #     colour = colours[i]
    #     light_colour = light_colours[i]
    #     axs[1].scatter(x[i], y[i], marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize_right)
    marker = markers[0]
    colour = colours[0]
    light_colour = light_colours[0]
    axs[1].scatter(x, y, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize_right)
    axs[1].plot(x_fit, y_fit, color=red, zorder=-1)
    # axs[1].scatter(0, intercept, marker='*', color=light_colour, edgecolors=colour, linewidths=edgewidth, s=25)#, label='y-intercept')
    axs[1].set_xlabel('$N^{-3/2}$')
    axs[1].set_ylabel('$E/N$ [K]')
    plt.tight_layout()

    output_filename = 'finite_size_effect_panel_plot_structure_factor_and_energy_vs_1-over-N32_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()





n1 = 0.067
n2 = 0.072
n_axis = np.array(n_axis)
E_infs = np.array(E_infs)
y1 = E_infs[np.where(np.isclose(n_axis, n1))]
y2 = E_infs[np.where(np.isclose(n_axis, n2))]
slope = (y2-y1)/(1/n2-1/n1)
print(f'Extrapolated P_c = {slope} KA^-2')

plt.plot(n_axis, E_infs, marker='o', color=red)
plt.show()





## Manual estimate of \delta P = (\delta E_N^S - \delta E_N^L) / (v_S - v_L) for N=80 (using n=0.067 and n=0.07 values)
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

n_l = 0.067
n_s = 0.07
delta_E_s = 0.11378941
delta_E_l = 0.04676058
v_s = 1/n_s # get_volume_per_particle(N=30, density=0.067)
v_l = 1/n_l # get_volume_per_particle(N=30, density=0.07)

delta_P = (delta_E_s - delta_E_l) / (v_s - v_l)
print(f'delta_P = {delta_P:.5f} KA^{-2}')
