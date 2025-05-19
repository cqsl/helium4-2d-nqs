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
    left, bottom, width, height = [0.64, 0.48, 0.3, 0.43] 
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
red = convert_rgb_255_to_1_scale((186, 30, 15))

light_blue = lighter(colors.to_rgb(blue), 0.7)
light_green = lighter(colors.to_rgb(green), 0.7)
light_yellow = lighter(colors.to_rgb(yellow), 0.7)

colours = [blue, yellow, green]
light_colours = [light_blue, light_yellow, light_green]

markers = ['o','^','s']






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




################################## PLOT ##################################
# plot_setup()
ax, ax_inset = plot_with_inset_setup()

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
n_f_inverse = 1/0.0673
n_m_inverse = 1/0.0698
ax.axvspan(n_f_inverse, n_m_inverse, alpha=0.2, color='gray', label='Obtained with the N=80 data')

## Add a line in the coexistence region with the slope (i.e. the equilibrium pressure) found in the isobaric ensemble
x1, y1 = 1/0.069, energies[np.where(np.isclose(densities, 0.069))]
x = np.linspace(1/0.08, 1/0.06, 100)
m = -0.485  # slope approximated using isobaric ensemble simulations
y = m * (x - x1) + y1
ax.plot(x, y, color=red)

# ## Take two points in the coexistence region to calculate the slope between them
# ## and compare it to the slope found in the isobaric ensemble
# y1 = energies[np.where(np.isclose(densities, 0.069))]
# y2 = energies[np.where(np.isclose(densities, 0.07))]
# print(f'slope = {(y2-y1)/(1/0.07-1/0.069)}')
# print(f'slope found using the isobaric ensemble = {m}')

ax.set_xlabel('$n^{-1} \ [\mathrm{\AA}^2]$')
ax.set_ylabel('$E/N$ [K]')

# Adjust the positions slightly to separate the labels more
n_m_inverse_adjusted = n_m_inverse - 0.1
n_f_inverse_adjusted = n_f_inverse + 0.1

## Set custom x-axis labels
ax.set_xticks([12, n_m_inverse_adjusted, n_f_inverse_adjusted, 16, 18, 20])
ax.set_xticklabels(['12', '$n_m^{-1}$', '$n_f^{-1}$', '16', '18', '20'])

## Remove the tick marks at the positions of n_m_inverse and n_f_inverse
ticks = ax.xaxis.get_major_ticks()
ticks[1].tick1line.set_visible(False)  # Remove tick mark at position 1
ticks[2].tick1line.set_visible(False)  # Remove tick mark at position 2

# Adjust font size for specific labels
for label in ax.get_xticklabels():
    if label.get_text() in ['$n_m^{-1}$', '$n_f^{-1}$']:
        label.set_fontsize(7)  # Set smaller font size for specific labels


ax.axvline(n_m_inverse, color='black', linestyle='--', linewidth=0.5)
ax.axvline(n_f_inverse, color='black', linestyle='--', linewidth=0.5)




################################################################################
#          Make inset of E/N vs n^{-1} but for N=80 instead of N=30            #
################################################################################


N = 80
working_dir = os.getcwd()
data_path = 'log_files/N=80/'
full_path = os.path.join(working_dir, data_path)

liquid_densities = []
liquid_energies = []
liquid_energies_err = []
solid_densities = []
solid_energies = []
solid_energies_err = []
metastable_flags = []

def extract_density(filename):
    return float(re.search(r'density=([\d.]+)', filename).group(1))

filenames = os.listdir(full_path)
filenames = [filename for filename in filenames if filename.endswith('.log')]
sorted_filenames = sorted(filenames, key=extract_density)


for filename in sorted_filenames:
    if filename.startswith('he4') and filename.endswith('.log') and f'N={N}' in filename: # filename.startswith('_he4') and 

        ## Extract the data from the filename
        phase = 'liquid' if 'liquid' in filename else 'solid'
        N = int(re.search(r'N=(\d+)', filename).group(1))
        density = float(re.search(r'density=([\d.]+)', filename).group(1))

        ## Load the data contained in the file
        with open(full_path + filename) as f:
            data_log = json.load(f)
        energy = np.array(data_log['Energy']['Mean'])
        energy_err = np.array(data_log['Energy']['Sigma'])
    
        mean_energy = alpha * np.mean(energy[-ns:]) / N  # [K]
        mean_energy_err = alpha * np.mean(energy_err[-ns:]) / N

        ## Store the results in the appropriate list
        if phase == 'liquid':
            liquid_densities.append(density)
            liquid_energies.append(mean_energy)
            liquid_energies_err.append(mean_energy_err)
        elif phase == 'solid':
            solid_densities.append(density)
            solid_energies.append(mean_energy)
            solid_energies_err.append(mean_energy_err)
        else:
            raise ValueError('Phase should be either liquid or solid!')

liquid_densities = np.array(liquid_densities)
liquid_energies = np.array(liquid_energies)
solid_densities = np.array(solid_densities)
solid_energies = np.array(solid_energies)


## Sort the inverse number densities and the associated energies for the spline function to work properly
inverse_liquid_densities = 1/liquid_densities
liquid_mask = np.argsort(inverse_liquid_densities)
inverse_liquid_densities = inverse_liquid_densities[liquid_mask]
sorted_liquid_energies = liquid_energies[liquid_mask]
inverse_solid_densities = 1/solid_densities
inverse_solid_densities = inverse_solid_densities
solid_mask = np.argsort(inverse_solid_densities)
inverse_solid_densities = inverse_solid_densities[solid_mask]
sorted_solid_energies = solid_energies[solid_mask]

spline_func_liquid = UnivariateSpline(inverse_liquid_densities, sorted_liquid_energies, k=4)
spline_func_solid = UnivariateSpline(inverse_solid_densities, sorted_solid_energies, k=4)
spline_func_liquid_derivative = spline_func_liquid.derivative()
spline_func_solid_derivative = spline_func_solid.derivative()

xs_l = np.linspace(np.min(inverse_liquid_densities)-0.3, np.max(inverse_liquid_densities), 300)
ax_inset.plot(xs_l, spline_func_liquid(xs_l), color=yellow, zorder=10)

xs_s = np.linspace(np.min(inverse_solid_densities), np.max(inverse_solid_densities)+0.3, 300)
ax_inset.plot(xs_s, spline_func_solid(xs_s), color=green, zorder=0)

## Get the freezing and melting number densities (n_f and n_m) by solving a system of two equations
def eqns(x):
    """ https://stackoverflow.com/questions/48362180/find-common-tangent-line-between-two-cubic-curves """
    f1 = spline_func_liquid
    f2 = spline_func_solid
    x1, x2 = x[0], x[1]
    eps = 0.0001
    df1 = (f1(x1+eps)-f1(x1-eps))/(2*eps)
    df2 = (f2(x2+eps)-f2(x2-eps))/(2*eps)
    eq1 = df1 - df2
    eq2 = df1 *(x1 - x2) - (f1(x1) - f2(x2))
    return [eq1, eq2]

n_f_guess = 0.068  # A^{-2}
n_m_guess = 0.071  # A^{-2}
initial_guess = [1/n_f_guess, 1/n_f_guess]
res = least_squares(eqns, initial_guess)
n_f_inverse, n_m_inverse = res.x


## Error propagation (test)
J = res.jac  ## extract the Jacobian matrix
cov_matrix = np.linalg.inv(J.T @ J)  ## compute the covariance matrix
errors = np.sqrt(np.diag(cov_matrix))  ## compute the standard errors (square root of the diagonal elements of the covariance matrix)
n_f = 1/n_f_inverse
n_m = 1/n_m_inverse
# Calculate the errors for the inverses
error_n_f_inverse, error_n_m_inverse = errors
error_n_f = error_n_f_inverse / n_f_inverse**2
error_n_m = error_n_m_inverse / n_m_inverse**2
print("n_f:", n_f, "±", error_n_f)
print("n_m:", n_m, "±", error_n_m)


print(f'Freezing density: {n_f:.5f} A^-2')
print(f'Melting density: {n_m:.5f} A^-2')
P_eq = -(spline_func_solid(n_m_inverse) - spline_func_liquid(n_f_inverse))/(n_m_inverse-n_f_inverse)
print(f'Equilibrium pressure: {P_eq:.5f} KA^-2')

pt1 = (n_f_inverse, spline_func_liquid(n_f_inverse))
pt2 = (n_m_inverse, spline_func_solid(n_m_inverse))
# ax_inset.axline(pt1, pt2, color='black', linestyle='dotted', linewidth=1, label='Isobaric line')


markersize = 2.5
markeredgewidth = 0.5

ax_inset.errorbar(
    1/liquid_densities, 
    liquid_energies, 
    yerr=liquid_energies_err,
    color=light_yellow, 
    ecolor=yellow,
    marker=markers[1], 
    markersize=markersize,
    markeredgecolor=yellow,
    markeredgewidth=markeredgewidth,
    linestyle='',
    label='Liquid',
    zorder=10,
)
ax_inset.errorbar(
    1/solid_densities, 
    solid_energies, 
    yerr=solid_energies_err,
    color=light_green, 
    ecolor=green,
    marker=markers[2], 
    markersize=markersize,
    markeredgecolor=green,
    markeredgewidth=markeredgewidth,
    linestyle='',
    label='Solid',
)
# ax_inset.axvspan(n_f_inverse, n_m_inverse, alpha=0.2, color='gray', label='Liquid-solid mixture')
# ax_inset.set_xlabel('$n^{-1} \ [\mathrm{\AA}^2]$', fontsize=6, labelpad=0.6)
# ax_inset.set_ylabel('$E/N$ [K]', fontsize=6, labelpad=0.6)
ax_inset.legend(fontsize=6)
ax_inset.set_xlim(13.86,14.8)
ax_inset.set_ylim(0.21,0.76)
ax_inset.tick_params(axis='both', which='major', labelsize=8,)# pad=0.5)
ax_inset.tick_params(axis='both', which='minor', labelsize=8,)# pad=0.5)
ax_inset.text(0.95, 0.95, '$N=80$', transform=ax_inset.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')



output_filename = f'energy_vs_density_phase_diagram_he4_N=30_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
# plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.show()