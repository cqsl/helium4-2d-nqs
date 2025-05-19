import os
import json
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit




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
markeredgewidth = 0.5
errorbarwidth = markeredgewidth
capsize = 2












## Manually tune the figure size
plt.figure(figsize=(3.75,2.5), dpi=200, facecolor='white')

N = 80
working_dir = os.getcwd()
# data_path = 'data_log/'
data_path = 'data_log_N=80/'
full_path = os.path.join(working_dir, data_path)
alpha = 1.37643  # hbar^2/(m*r_m^2*k_B) [K]
ns = 1024  # number of values to consider for the average


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
        
        # ## Plot energy minimization curve
        # plt.plot(range(energy.size), energy)
        # plt.title(f'N={N}, density={density}, phase={phase}')
        # plt.show()

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


all_densities = liquid_densities + solid_densities
all_energies = liquid_energies + solid_energies
indices = {density: index for index, density in enumerate(all_densities)}
unique_densities = []
unique_energies = []
for index, density in enumerate(all_densities):
    if indices[density] == index:
        unique_densities.append(density)
        unique_energies.append(all_energies[index])
all_densities = np.array(unique_densities)
all_energies = np.array(unique_energies)



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
plt.plot(xs_l, spline_func_liquid(xs_l), color=blue, zorder=0)

xs_s = np.linspace(np.min(inverse_solid_densities), np.max(inverse_solid_densities)+0.3, 300)
plt.plot(xs_s, spline_func_solid(xs_s), color=green, zorder=0)





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
plt.axline(pt1, pt2, color='black', linestyle='dotted', linewidth=1, label='Isobaric line')




plt.errorbar(
    1/liquid_densities, 
    liquid_energies, 
    yerr=liquid_energies_err,
    color=light_blue, 
    ecolor=blue,
    marker=markers[0], 
    markersize=3,
    markeredgecolor=blue,
    markeredgewidth=0.7,
    linestyle='',
    label='Liquid branch',
)
plt.errorbar(
    1/solid_densities, 
    solid_energies, 
    yerr=solid_energies_err,
    color=light_green, 
    ecolor=green,
    marker=markers[1], 
    markersize=3,
    markeredgecolor=green,
    markeredgewidth=0.7,
    linestyle='',
    label='Solid branch',
)
# plt.errorbar(1/liquid_densities, liquid_energies, yerr=liquid_energies_err, fmt='o', label='Liquid branch')
# plt.errorbar(1/solid_densities, solid_energies, yerr=solid_energies_err, fmt='o', label='Solid branch')
# plt.axvspan(1/0.0680, 1/0.0711, alpha=0.2, color='gray', label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
plt.axvspan(n_f_inverse, n_m_inverse, alpha=0.2, color='gray', label='Liquid-solid mixture')


plt.xlabel('$n^{-1} \ [\mathrm{\AA}^2]$')
plt.ylabel('$E/N$ [K]')
plt.legend()

plt.xlim(13.86,14.8)
plt.ylim(0.21,0.76)
output_filename = 'phase_diagram_he4_N=80_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
# plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()