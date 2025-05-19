import os, re
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
green = convert_rgb_255_to_1_scale((0,153,68))
yellow = convert_rgb_255_to_1_scale((255,187,0))

light_blue = lighter(colors.to_rgb(blue), 0.7)
light_green = lighter(colors.to_rgb(green), 0.7)
light_yellow = lighter(colors.to_rgb(yellow), 0.7)

colours = [blue, yellow, green]
light_colours = [light_blue, light_yellow, light_green]

markers = ['o','^','s','D']

markersize = 8
edgewidth = 0.5





plot_setup()

##  We assume R_axis is the same for all files in a folder

Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}

paths = [
    'entropy_data/N=30/Rn12=2.39_changing_origins/',
    'entropy_data/N=80/Rn12=2.39_changing_origins/',
    'entropy_data/N=80/Rn12=3.34_changing_origins/',
]



for i, path_to_data in enumerate(paths):
    entropy_files = [file for file in os.listdir(path_to_data) if file.startswith('entropy_') and file.endswith('.npz')]
    Rn12 = float(re.search(r'Rn12=([\d.]+)', path_to_data).group(1))
    N = int(re.search(r'N=(\d+)', path_to_data).group(1))

    densities = []
    entropies = []
    entropies_err = []

    for f in entropy_files:
        density = float(re.search(r'density=(\d*\.\d+)', f).group(1))

        data = np.load(os.path.join(path_to_data, f))
        assert data["density"] == density
        R_axis = data['R_axis']  # (num_R=1,)
        entropy_mean = data['entropy']  # (num_R,)
        entropy_err = data['entropy_err']  # (num_R,)
        
        densities.append(density)
        entropies.append(entropy_mean)
        entropies_err.append(entropy_err)

    marker = markers[0]
    colour = blue
    light_colour = light_blue

    # plt.errorbar(
    #     densities, 
    #     entropies, 
    #     yerr=entropies_err,
    #     color=light_colour, 
    #     ecolor=colour,
    #     marker=marker,
    #     markersize=3,
    #     markeredgecolor=colour,
    #     markeredgewidth=0.7,
    #     linestyle='',
    # )
    marker = markers[i]
    light_colour = light_colours[i]
    colour = colours[i]


    Mx = Mxs[f'N={N}']
    My = Mys[f'N={N}']
    a = np.sqrt(2 / (np.sqrt(3) * density))  # lattice constant, obtained by fixing n=N/(Lx*Ly) [A]
    # lx x ly defines a rectangle that contains two unit cells (in terms of its area)
    lx = a  # [A]
    ly = np.sqrt(3) * a  # [A]
    Lx = Mx * lx  # [A]
    Ly = My * ly  # [A]
    V = Lx * Ly

    C = Rn12  # fixed 
    R = C/density**(1/2)
    N_tot = (C/R)**2 * V  # Rn^{1/2}=C ==> R(N/V)^{1/2}=C ==> N=(C/R)^2 V
    N_avg = C**2 * np.pi  # N_avg=n*A_R=n*(pi*R^2)=n*pi*(C/n^{1/2})^2=C^2*pi
    print(f'N_tot={N_tot} should be equal to N={N}')
    print(f'density={density}, R={R}, N_avg={N_avg}')


    # label = '$Rn^{1/2}=$' + f'\ ${Rn12}$'
    label = f'$N_\mathrm{{A}} \\approx {round(N_avg)}$'
    # if i == 0 or i == 1:
    label += f', $N={N}$'
    plt.scatter(densities, entropies, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=label)


plt.axvspan(0.06811, 0.07159, alpha=0.2, color='gray')# label='Obtained with the N=80 data')
# plt.axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
# plt.title(f'$N={N}$,' + '$\ R n^{1/2} =$' + f'{R_axis[radius_index]:.3f}' + '$\ \mathrm{\AA}$')
plt.xlabel('$n \ [\mathrm{\AA}^{-2}]$')
plt.ylabel('$S_2$')
plt.legend(fontsize=6.5)
plt.tight_layout()

output_filename = f's2_vs_density_he4_N={N}_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# plt.savefig(output_filename, dpi=400, bbox_inches='tight')
plt.show()





# ##########################################
# #      Check geometry of partition A     #
# ##########################################

# densities = [0.05, 0.09]
# for density in densities:

#     N = 80  # number of helium-4 atoms
#     d = 2  # spatial dimension
#     Mx = 8
#     My = 5

#     a = np.sqrt(2 / (np.sqrt(3) * density))  # lattice constant, obtained by fixing rho=N/(Lx*Ly) [A]
#     # lx x ly defines a rectangle that contains two unit cells (in terms of its area)
#     lx = a  # [A]
#     ly = np.sqrt(3) * a  # [A]
#     Lx = Mx * lx  # [A]
#     Ly = My * ly  # [A]
#     V = Lx * Ly

#     C = 2.38793318
#     R = C/density**(1/2)
#     N_tot = (C/R)**2 * V  # Rn^{1/2}=C ==> R(N/V)^{1/2}=C ==> N=(C/R)^2 V
#     N_avg = C**2 * np.pi  # N_avg=n*A_R=n*(pi*R^2)=n*pi*(C/n^{1/2})^2=C^2*pi
#     print(f'N_tot={N_tot} should be equal to N={N}')
#     print(f'density={density}, R={R}, N_avg={N_avg}')

