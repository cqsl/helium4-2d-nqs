import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import re
import os
from scipy.signal import find_peaks



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
    fig, axs = plt.subplots(1, num_subplots, figsize=(50*width,height), dpi=200, facecolor='white')
    return fig, axs


global_setup(columns='twocolumn', paper='a4paper', fontsize=10)




N = 80
rm = 2.9673  # [A]
path = f'g2_data/N={N}/'

def extract_density(filename):
    return float(re.search(r'density=(\d+\.\d+)', filename).group(1))
filenames = os.listdir(path)
filenames = [f for f in filenames if f.startswith('pair_correlation_function') and f'N={N}' in f and f.endswith('.npz')]
# filenames = [f for f in filenames if ('density=0.085' in f or 'density=0.090' in f)]
filenames = [f for f in filenames if 0.069 <= extract_density(f) <= 0.09]
sorted_filenames = sorted(filenames, key=extract_density)

fig, axs = plt.subplots(1, len(sorted_filenames), figsize=(15,3))


for i, filename_head in enumerate(sorted_filenames):
        
    # N = int(re.search(r'N=(\d+)', filename_head).group(1))
    density = extract_density(filename_head)

    ## Load data
    filename_path = os.path.join(path, filename_head)
    data = jnp.load(filename_path)
    num_coords = data['num_coords']
    Lx = data['Lx']
    Ly = data['Ly']
    rs = data['rs']
    estimators = data['estimators']

    area = Lx * Ly  # [A^2]
    estimators /= (area * density**2 * rm**2)  # [dimensionless]
    estimators_2d = estimators.reshape(num_coords)
    assert num_coords[0] == num_coords[1]
    size = num_coords[0]

    rs *= rm  # [A]
    rs_2d = rs.reshape(*num_coords,-1)
    x = rs_2d[:,size//2,0][-size//2:]  # (size,size,d) -> (size//2,)
    y = estimators_2d[:,size//2][-size//2:]  # (size,size) -> (size//2,)
    # y -= 1  # g2(x=0,y) - 1
    




    #################################
    #  Fit the decay of the maxima  #
    #################################

    def find_local_minima_maxima(data):
        derivative = np.diff(data)
        sign_changes = np.sign(derivative[:-1]) != np.sign(derivative[1:])
        minima_indices= np.where((sign_changes) & (derivative[:-1] < 0))[0] + 1
        maxima_indices = np.where((sign_changes) & (derivative[:-1] > 0))[0] + 1
        return minima_indices, maxima_indices

    minima_indices, maxima_indices = find_local_minima_maxima(y)
    x_local_maxima = x[maxima_indices]  
    y_local_maxima = y[maxima_indices]

    from scipy.optimize import curve_fit
    def decay_function(x, a, b, Ly):
        return a + b * (1/x + 1/(Ly - x))
    L_min = np.min((Lx, Ly))
    L_half = L_min / 2.
    wrapper_decay_function = lambda x, a, b: decay_function(x, a, b, L_min)
    popt, pcov = curve_fit(wrapper_decay_function, x_local_maxima, y_local_maxima)
    a, b = popt
    
    x_fit = np.linspace(2., L_half, 100)
    fitted_y = wrapper_decay_function(x_fit, *popt) 
    axs[i].plot(x_fit, abs(fitted_y-a)-1, color='blue', linestyle='--', label='Fitted Decay Function')
    axs[i].scatter(x_local_maxima, abs(y_local_maxima-a)-1, color='red', marker='*', s=10, zorder=10) # , label='First Local Maximum'




    ##########################################
    #  Find 1st max and plot r^{-1/3} curve  #
    ##########################################
    peaks, _ = find_peaks(y, prominence=0.01)
    positive_peaks = [peak for peak in peaks if y[peak] > 0]
    if peaks.size > 0:
        first_peak_index = peaks[0]
        first_peak_x = x[first_peak_index]
        first_peak_y = y[first_peak_index]

        # Plot the power law line 1/r^{1/3} starting from the first local maximum
        r_values = np.linspace(2., max(x), 100)
        power_law_values = first_peak_y * (first_peak_x / r_values)**(1/3)
        # axs[i].plot(r_values, power_law_values, color='red', linestyle='--', label='$\sim 1/y^{1/3}$')

        # Add a red star on the local maximum identified
        # axs[i].scatter(first_peak_x, first_peak_y, color='red', marker='*', s=10, zorder=10) # , label='First Local Maximum'



    axs[i].plot(x, abs(y-a)-1)
    axs[i].set_xlabel('$y \ [\mathrm{\AA}]$')
    axs[i].set_title(f'$n={density}$' + '$\ \mathrm{\AA}^{-2}$')

axs[0].set_ylabel('$|g_2(x=0,y)-a|-1$')

plt.tight_layout()

output_filename = f'g2_minus1_vs_r_with_1-r-1-3_vs_density_he4_N={N}_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# plt.savefig(output_filename, dpi=400, bbox_inches='tight')
plt.show()
