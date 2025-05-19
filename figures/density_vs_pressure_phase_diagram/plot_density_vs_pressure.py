import os
import netket as nk
from trial_wavefunction import He4JastrowLxLyParams
import re 
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
import flax
import json
import numpy as np


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
    plt.style.use(['seaborn-v0_8-white','quantum-plots.mplstyle'])
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
    left, bottom, width, height = [0.51, 0.51, 0.38, 0.35] 
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
red = convert_rgb_255_to_1_scale((186,30,15))

light_blue = lighter(colors.to_rgb(blue), 0.7)

markers = ['o','s']




########################### GET THE DATA ###########################
N = 30
alphaHe4 = 1.37643  
ns = 4096

def extract_pressure(filename):
    # Extract the pressure value from the filename
    return float(re.search('p=([\d\.]+)', filename).group(1))

res = {phase: {
    'pressure': [],
    'number_density': [],
    'number_density_err': [],
    'gibbs_energy': [],
    'gibbs_energy_err': []
    } for phase in ['liquid', 'solid']}

filenames = os.listdir(f'data_log/')
filenames = [filename for filename in filenames if filename.endswith('.log') or filename.endswith('.mpack')]
sorted_filenames = sorted(filenames, key=extract_pressure)


for filename in sorted_filenames:
    if filename.startswith('he4') and filename.endswith('.log'):

        filename_path = os.path.join(f'data_log/', filename)
        phase = 'liquid' if 'liquid' in filename else 'solid'  # extract phase
        pressure = extract_pressure(filename)  # extract pressure

        ## Load the energy data
        with open(filename_path) as f:
            data_log = json.load(f)
        energy = np.array(data_log['Energy']['Mean'])
        energy_err = np.array(data_log['Energy']['Sigma'])
        mean_energy = alphaHe4 * np.mean(energy[-ns:]) / N  # [K]
        mean_energy_err = alphaHe4 * np.mean(energy_err[-ns:]) / N  # [K]
        res[phase]['gibbs_energy'].append(mean_energy)
        res[phase]['gibbs_energy_err'].append(mean_energy_err)

        # ## Plot energy optimization curve
        # print(f'phase={phase}, P={pressure} KA^-2, E_mean={np.mean(energy[-ns:])}')
        # fig2 = plt.figure()
        # plt.plot(range(len(energy)), energy)
        # plt.title(f'phase: {phase}, P={pressure} KA^-2')
        # plt.show()
    
        densities = np.array(data_log['n']['value'][-ns:])
        number_density = np.mean(densities)
        number_density_err = np.std(densities) / np.sqrt(ns)
        # plt.plot(range(len(energy)), densities)
        # plt.ylabel('Density')
        # plt.title(f'phase: {phase}, P={pressure} KA^-2')
        # plt.show()

        res[phase]['pressure'].append(pressure)
        res[phase]['number_density'].append(number_density)
        res[phase]['number_density_err'].append(number_density_err)



################################## PLOT ##################################
plot_setup()

## Liquid branch
plt.errorbar(
    res['liquid']['pressure'][:-1], 
    res['liquid']['number_density'][:-1], 
    yerr=res['liquid']['number_density_err'][:-1],
    color=light_blue, 
    marker=markers[0],
    markersize=3,
    markeredgecolor=blue,
    markeredgewidth=0.7,
    linestyle='',
    label='VMC',
)
## Metastable liquid branch
plt.errorbar(
    res['liquid']['pressure'][-1:], 
    res['liquid']['number_density'][-1:], 
    yerr=res['liquid']['number_density_err'][-1:],
    color='white', 
    marker=markers[0],
    markersize=3,
    markeredgecolor=blue,
    markeredgewidth=0.7,
    linestyle='',
    label='VMC',
)
## Solid branch
plt.errorbar(
    res['solid']['pressure'][1:], 
    res['solid']['number_density'][1:], 
    yerr=res['solid']['number_density_err'][1:],
    color=light_blue, 
    marker=markers[0],
    markersize=3,
    markeredgecolor=blue,
    markeredgewidth=0.7,
    linestyle='',
    label='VMC',
)
## Metastable solid branch
plt.errorbar(
    res['solid']['pressure'][:1], 
    res['solid']['number_density'][:1], 
    yerr=res['solid']['number_density_err'][:1],
    color='white', 
    marker=markers[0],
    markersize=3,
    markeredgecolor=blue,
    markeredgewidth=0.7,
    linestyle='',
    label='VMC',
)
P_c = 0.485  # critical pressure [K/A^2]
delta_P_c = 0.005  # uncertainty in critical pressure [K/A^2]
plt.axvline(P_c, color=red)
plt.axvspan(P_c - delta_P_c, P_c + delta_P_c, color=red, alpha=0.1)

n_f_approx = 0.0673
n_m_appox = 0.0698
plt.axhline(n_f_approx, color='black', linestyle='--', linewidth=0.5)
plt.axhline(n_m_appox, color='black', linestyle='--', linewidth=0.5)

# Set custom x-axis and y-axis labels
plt.xticks(
    [0.40, 0.45, P_c, 0.52, 0.56, 0.60],
    ['0.40', '0.45', '$P_c$', '0.52', '0.56', '0.60']
)
plt.yticks(
    [0.066, n_f_approx, n_m_appox, 0.072],
    ['0.066', '$n_f$', '$n_m$', '0.072']
)

plt.xlabel('$P$ [K$\mathrm{\AA}^{-2}$]')
plt.ylabel('$n$ [$\mathrm{\AA}^{-2}$]')
# plt.legend()

output_filename = 'density_vs_pressure_phase_diagram_he4_N=30_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.tight_layout()
plt.savefig(output_filename, dpi=200, bbox_inches='tight')
plt.show()