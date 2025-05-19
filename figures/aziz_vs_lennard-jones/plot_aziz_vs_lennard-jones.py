import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors




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
green = convert_rgb_255_to_1_scale((0,153,68))

light_blue = lighter(colors.to_rgb(blue), 0.7)
light_green = lighter(colors.to_rgb(green), 0.7)

colours = [blue, green]
light_colours = [light_blue, light_green]

markers = ['o','s']

markersize = 8
edgewidth = 0.6




# Define constants for the Aziz potential
epsilon = 10.8  # [K]
A = 0.544850e6  # [dimensionless]
alpha = 13.353384  # [dimensionless]
C6 = 1.3732412  # [dimensionless]
C8 = 0.4253785  # [dimensionless]
C10 = 0.178100  # [dimensionless]
D = 1.241314  # [dimensionless]
r_m = 2.9673  # [Ang] (minimum distance)

# Define constants for the Lennard-Jones potential
epsilon_LJ = 10.22  # [K]
sigma = 2.556  # [Ang]

def aziz_potential(r):
    x = r / r_m
    F_x = np.where(x < D, np.exp(-(D/x - 1)**2), 1.0)
    term_exp = A * np.exp(-alpha * x)
    term_dispersion = (C6 / x**6 + C8 / x**8 + C10 / x**10) * F_x
    v_aziz = epsilon * (term_exp - term_dispersion)
    return v_aziz

def lj_potential(r):
    v_lj = 4 * epsilon_LJ * ((sigma / r)**12 - (sigma / r)**6)
    return v_lj

r = np.linspace(2.0, 8.0, 1000)  # [A]

v_aziz = aziz_potential(r)
v_lj = lj_potential(r)


## Manually tune the figure size
plt.figure(figsize=(3.75,2.5), dpi=200, facecolor='white')

plt.plot(r, v_aziz, label="Aziz", color=colours[0])
plt.plot(r, v_lj, label="Lennard-Jones", color=colours[1], linestyle='--')

plt.xlabel('$r \ [\mathrm{\AA}]$')
plt.ylabel('$V(r)$ [K]')
plt.axhline(0, color='black',linewidth=1.)
plt.ylim(-12., 16.)
plt.xlim(2.4, 4.6)
plt.legend()

output_filename = 'aziz_vs_lennard-jones.pdf'
plt.tight_layout()
# plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()



# ## Manually tune the figure size
# plt.figure(figsize=(3.75,2.5), dpi=200, facecolor='white')

# plt.plot(r, v_aziz, label="Aziz", color=colours[0], linewidth=2.)
# plt.plot(r, v_lj, label="Lennard-Jones", color=colours[1], linestyle='--', linewidth=2.)
# plt.xlabel('$r \ [\mathrm{\AA}]$', fontsize=14)
# plt.ylabel('$V(r)$ [K]', fontsize=14)
# plt.axhline(0, color='black', linewidth=1.)
# plt.ylim(-12., 16.)
# plt.xlim(2.4, 4.6)
# plt.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# output_filename = 'aziz_vs_lennard-jones_for_slides.pdf'
# plt.tight_layout()
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()
