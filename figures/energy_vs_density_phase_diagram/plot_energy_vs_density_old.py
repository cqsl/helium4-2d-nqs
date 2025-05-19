import os
import json
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline


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
    left, bottom, width, height = [0.6, 0.57, 0.35, 0.35] 
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
red = convert_rgb_255_to_1_scale((186, 30, 15))

light_blue = lighter(colors.to_rgb(blue), 0.7)

markers = ['o','s']




########################### GET THE DATA ###########################
working_dir = os.getcwd()
N = 30
alpha = 1.37643  # hbar^2/(m*r_m^2*k_B) [K]
ns = 1024  # number of values to consider for the average

data_path = f'log_files/N={N}/'
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
        # N = int(re.search(r'N=(\d+)', filename).group(1))
        density = float(re.search(r'density=([\d.]+)', filename).group(1))

        ## Load the data contained in the file
        with open(full_path + filename) as f:
            data_log = json.load(f)
        energy = np.array(data_log['Energy']['Mean'])
        energy_err = np.array(data_log['Energy']['Sigma'])
        
        # ## Plot energy minimization curve
        # plt.plot(range(energy.size), energy)
        # plt.title(f'N={N}, density={density}')
        # plt.show()

        mean_energy = alpha * np.mean(energy[-ns:]) / N  # [K]
        mean_energy_err = alpha * np.mean(energy_err[-ns:]) / N

        ## Store the results in the appropriate list
        densities.append(density)
        energies.append(mean_energy)
        energies_err.append(mean_energy_err)

densities = np.array(densities)
energies = np.array(energies)




################################## PLOT ##################################
# plot_setup()
ax, ax_inset = plot_with_inset_setup()

# ## Fit a straight line to the points in the (approximated) coexistence region
# inv_den_min = 14.2
# inv_den_max = 14.9
# mask = (1/densities >= inv_den_min) & (1/densities <= inv_den_max)
# x = 1/densities[mask]
# y = energies[mask]
# coefficients = np.polyfit(x, y, 1)
# print(f'Linear fit parameters: {coefficients}')
# fit_line = np.poly1d(coefficients)
# fit_axis = np.linspace(inv_den_min-1, inv_den_max+1, 100)
# ax.plot(fit_axis, fit_line(fit_axis), color=red, zorder=0, label='Linear fit')
# ax_inset.plot(fit_axis, fit_line(fit_axis), color=red, zorder=0, label='Linear fit')


# ## Spline fit of all the data
# from scipy.interpolate import UnivariateSpline
# inverse_densities = 1/densities
# mask = np.argsort(inverse_densities)
# inverse_densities = inverse_densities[mask]
# sorted_energies = energies[mask]
# spline_func = UnivariateSpline(inverse_densities, sorted_energies, k=5)
# spline_axis = np.linspace(np.min(inverse_densities), np.max(inverse_densities), 300)
# ax.plot(spline_axis, spline_func(spline_axis), color='tab:blue', label='Spline fit', zorder=0)
# ax_inset.plot(spline_axis, spline_func(spline_axis), color='tab:blue', label='Spline fit', zorder=0)

ax.errorbar(
    1/densities, 
    energies, 
    yerr=energies_err,
    color=light_blue, 
    ecolor=blue,
    marker=markers[0], 
    markersize=3,
    markeredgecolor=blue,
    markeredgewidth=0.7,
    linestyle='',
    label='VMC',
)
ax_inset.errorbar(
    1/densities, 
    energies, 
    yerr=energies_err,
    color=light_blue, 
    ecolor=blue,
    marker=markers[0], 
    markersize=3,
    markeredgecolor=blue,
    markeredgewidth=0.7,
    linestyle='',
    label='VMC',
)
# ax.axvspan(1/0.0680, 1/0.0711, alpha=0.2, color='gray', label='Liquid-solid coexistence\n[Gordillo \& Ceperley 1998]')
# ax_inset.axvspan(1/0.0680, 1/0.0711, alpha=0.2, color='gray', label='Liquid-solid coexistence\n[Gordillo \& Ceperley 1998]')
n_f_inverse = 1/0.0673
n_m_inverse = 1/0.0698
ax.axvspan(n_f_inverse, n_m_inverse, alpha=0.2, color='gray', label='Obtained with the N=80 data')
ax_inset.axvspan(n_f_inverse, n_m_inverse, alpha=0.2, color='gray', label='Obtained with the N=80 data')

## Add a line in the coexistence region with the slope (i.e. the equilibrium pressure) found in the isobaric ensemble
x1, y1 = 1/0.069, energies[np.where(np.isclose(densities, 0.069))]
x = np.linspace(1/0.08, 1/0.06, 100)
m = -0.485  # slope approximated using isobaric ensemble simulations
y = m * (x - x1) + y1
ax.plot(x, y, color=red)
ax_inset.plot(x, y, color=red)


## Take two points in the coexistence region to calculate the slope between them
## and compare it to the slope found in the isobaric ensemble
y1 = energies[np.where(np.isclose(densities, 0.069))]
y2 = energies[np.where(np.isclose(densities, 0.07))]
print(f'slope = {(y2-y1)/(1/0.07-1/0.069)}')
print(f'slope found using the isobaric ensemble = {m}')



ax.set_xlabel('$n^{-1} \ [\mathrm{\AA}^2]$')
ax.set_ylabel('$E/N$ [K]')

ax_inset.set_xlim(13.8,15.4)
ax_inset.set_ylim(-0.05,0.8)

# Adjust the positions slightly to separate the labels more
n_m_inverse_adjusted = n_m_inverse - 0.1
n_f_inverse_adjusted = n_f_inverse + 0.1



## Set custom x-axis labels
ax.set_xticks([12, n_m_inverse_adjusted, n_f_inverse_adjusted, 16, 18, 20])
ax.set_xticklabels(['12', '$n_\mathrm{m}^{-1}$', '$n_\mathrm{f}^{-1}$', '16', '18', '20'])

## Remove the tick marks at the positions of n_m_inverse and n_f_inverse
ticks = ax.xaxis.get_major_ticks()
ticks[1].tick1line.set_visible(False)  # Remove tick mark at position 1
ticks[2].tick1line.set_visible(False)  # Remove tick mark at position 2

# Adjust font size for specific labels
for label in ax.get_xticklabels():
    if label.get_text() in ['$n_\mathrm{m}^{-1}$', '$n_\mathrm{f}^{-1}$']:
        label.set_fontsize(7)  # Set smaller font size for specific labels


ax.axvline(n_m_inverse, color='black', linestyle='--', linewidth=0.5)
ax.axvline(n_f_inverse, color='black', linestyle='--', linewidth=0.5)

ax_inset.axvline(n_m_inverse, color='black', linestyle='--', linewidth=0.5)
ax_inset.axvline(n_f_inverse, color='black', linestyle='--', linewidth=0.5)

## Set custom x-axis labels in the inset as well
ax_inset.set_xticks([n_m_inverse, n_f_inverse])
ax_inset.set_xticklabels(['$n_\mathrm{m}^{-1}$', '$n_\mathrm{f}^{-1}$'])



# plt.legend()
# plt.xlim(14,14.8)
# plt.ylim(0.25,0.7)
output_filename = f'energy_vs_density_phase_diagram_he4_N={N}_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
# plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.show()