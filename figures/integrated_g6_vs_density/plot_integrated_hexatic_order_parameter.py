import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
import numpy as np
import re, os
from jax.scipy.integrate import trapezoid





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








markersize = 8
markeredgewidth = 0.5
errorbarwidth = markeredgewidth
capsize = 2



Mxs = {'N=30': 5, 'N=80': 8}
Mys = {'N=30': 3, 'N=80': 5}
rm = 2.9673  # [A]
fig = plot_setup()

N_axis = [30,80]

for i, N in enumerate(N_axis):
    path_to_data = f'hexatic_data/N={N}/'
    # if N == 30:
    # density_folders = sorted([folder for folder in os.listdir(path_to_data) if folder.startswith('density=') and '1SPC' in folder])
    # elif N == 80:
    density_folders = sorted([folder for folder in os.listdir(path_to_data) if folder.startswith('density=')])

    densities = []
    integrated_g6s = []
    integrated_g6s_err = []

    for folder in density_folders:
        density = float(re.search(r'density=([\d.]+)', folder).group(1))
        
        npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]
        
        bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)
        
        for file in npz_files:
            data = np.load(os.path.join(path_to_data, folder, file))

            a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
            lx = a  # [A]
            ly = np.sqrt(3)*a  # [A]
            Mx = Mxs[f'N={N}']
            My = Mys[f'N={N}']
            Lx = Mx*lx  # [A]
            Ly = My*ly  # [A]
            L = jnp.array([Lx,Ly]) / rm  # [dimensionless]
            volume = jnp.prod(L)  # [dimensionless] 

            ## Load data
            num_r = data['num_r']
            radius_axis = data['radius_axis']  # [dimensionless]
            estimators = data['estimators']

            mask = ~np.isnan(estimators)
            # integrated_g6 = (2 * np.pi / volume) * np.sum(np.real(estimators[mask]) * radius_axis[mask])  # INCORRECTLY assumes a unit spacing between the radii
            dr = radius_axis[1] - radius_axis[0]  # [dimensionless]
            integrated_g6 = (2 * np.pi / volume) * np.sum(np.real(estimators[mask]) * radius_axis[mask] * dr)
            # integrated_g6 = (2 * np.pi / volume) * trapezoid(y=np.real(estimators[mask]) * radius_axis[mask], x=radius_axis[mask])  # <-- the above is equivalent to this
            bootstrap_samples.append(integrated_g6)  # add one bootstrap sample

        bootstrapped_G6 = np.mean(bootstrap_samples, axis=0)  # (num_R,)
        bootstrapped_G6_err = np.std(bootstrap_samples, axis=0) / np.sqrt(len(npz_files)) # (num_R,)

        densities.append(density)
        integrated_g6s.append(bootstrapped_G6)
        integrated_g6s_err.append(bootstrapped_G6_err)

    #     plt.scatter(len(bootstrap_samples)*[density], bootstrap_samples)
    # plt.show()

    marker = markers[i]
    colour = colours[i]
    light_colour = light_colours[i]
    # plt.errorbar(
    #     densities,
    #     integrated_g6s,
    #     yerr=integrated_g6s_err,
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
    plt.scatter(densities, integrated_g6s, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=f'$N={N}$')

plt.axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
plt.xlabel('$n \ [\mathrm{\AA}^{-2}]$')
plt.ylabel('$G_6$')  # '$2 \pi \int g_6(r) r \ dr / V$'
plt.legend()
plt.yscale('log')
plt.tight_layout()

output_filename = 'integrated_g6_vs_density_he4_vs_N=30-80_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()
