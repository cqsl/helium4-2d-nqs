import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams, colors
import re, os



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

def plot_setup_2_subplots_4_insets(aspect_ratio=1/1.62, width_ratio=1., wide=False, num_subplots=2): 
    """ Two vertical subplots, with 4 insets: 2 insets in the top and 2 in the bottom subplots. """
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio

    #### MIGHT NEED TO TUNE THE `figsize` ARGUMENT ####
    fig, axs = plt.subplots(num_subplots, 1, figsize=(width, 1.7 * height), dpi=200, facecolor='white')
    
    # Define the size for the insets (make them of the same size)
    inset_width, inset_height = 0.15, 0.15

    # Top left inset
    left, bottom = 0.3, 0.67
    ax_inset_top_left = fig.add_axes([left, bottom, inset_width, inset_height])

    # Top right inset
    left, bottom = 0.78, 0.59
    ax_inset_top_right = fig.add_axes([left, bottom, inset_width, inset_height]) 

    # Bottom left inset
    left, bottom = 0.29, 0.34
    ax_inset_bottom_left = fig.add_axes([left, bottom, inset_width, inset_height])

    # Bottom right inset
    left, bottom = 0.75, 0.25
    ax_inset_bottom_right = fig.add_axes([left, bottom, inset_width, inset_height])   
    
    axs_insets = [ax_inset_top_left, ax_inset_top_right, ax_inset_bottom_left, ax_inset_bottom_right]
    return fig, axs, axs_insets

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
edgewidth = 0.6








fig, axs, axs_inset = plot_setup_2_subplots_4_insets(num_subplots=2)
ax_inset_top_left, ax_inset_top_right, ax_inset_bottom_left, ax_inset_bottom_right = axs_inset
rm = 2.9673  # [A]
Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}
N_axis = [30,56,80]

for i, N in enumerate(N_axis):
    path_to_data = f'structure_factor_data/N={N}/'

    density_folders = sorted([folder for folder in os.listdir(path_to_data) if folder.startswith('density=')])

    densities = []
    struct_fact_peaks = []
    struct_fact_peaks_err = []

    for folder in density_folders:
        density = float(re.search(r'density=([\d.]+)', folder).group(1))
        
        npz_files = [f for f in os.listdir(os.path.join(path_to_data, folder)) if f.endswith('.npz')]
        
        bootstrap_samples = []  # store bootstrap (entropy) samples (for a given density)
        
        for file in npz_files:
            data = np.load(os.path.join(path_to_data, folder, file))

            ## Load data
            n_max = data['n_max']
            ks = data['ks']/rm  # [A^-1]
            estimators = data['estimators']
            


            ## Get the peak at the reciprocal lattice vector, i.e. S(G). This corresponds to
            ## the 2nd largest value after S(0)
            S_G = jnp.sort(jnp.real(estimators))[-2]
            print(f'SG1 = {S_G}')


            
            ## Get S(G) for a specific G (for instance G=b1+b2, where bi are the reciprocal space basis vectors)
            ## Define physical and wavefunction variables
            d = 2  # spatial dimension
            Mx = Mxs[f'N={N}']
            My = Mys[f'N={N}']
            a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
            lx = a  # [A]
            ly = np.sqrt(3)*a  # [A]
            Lx = Mx*lx  # [A]
            Ly = My*ly  # [A]
            L = np.array([Lx,Ly]) / rm  # [dimensionless]
            
            ## Define the direct space basis vectors
            a1 = a * jnp.array([1.,0.])
            a2 = a * jnp.array([1/2,np.sqrt(3)/2]) 
            direct_basis = jnp.column_stack([a1,a2])  # [A]

            ## Obtain the associated reciprocal space basis vectors
            reciprocal_basis = 2 * jnp.pi * jnp.linalg.inv(direct_basis)  # [A^-1] B=2*pi*A^{-1}
            b1, b2 = reciprocal_basis
            print(f'b1={b1}')
            print(f'b2={b2}')
            b1 = 2 * jnp.pi / a * jnp.array([1.,-1/np.sqrt(3)])
            b2 = 4 * jnp.pi / (a * jnp.sqrt(3.)) * jnp.array([0.,1.])
            print(f'b1={b1}')
            print(f'b2={b2}')
            G = b2
            tolerance = 1e-5
            diff = np.linalg.norm(ks - G, axis=1)
            position = np.where(diff < tolerance)[0]
            S_G = jnp.real(estimators[position])
            print(f'SG2 = {S_G}')
            print('==-========')




            
            bootstrap_samples.append(S_G)  # add one bootstrap sample

        bootstrapped_S_G = np.mean(bootstrap_samples, axis=0)  # (num_R,)
        bootstrapped_S_G_err = np.std(bootstrap_samples, axis=0) / np.sqrt(len(npz_files)) # (num_R,)

        densities.append(density)
        struct_fact_peaks.append(bootstrapped_S_G)
        struct_fact_peaks_err.append(bootstrapped_S_G_err)

    marker = markers[i]
    colour = colours[i]
    light_colour = light_colours[i]
    axs[0].scatter(densities, np.array(struct_fact_peaks) / N, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=f'$N={N}$')

axs[0].axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
axs[0].set_xticks([])
axs[0].set_ylabel('$S(\mathbf{G}) / N$')
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.28), ncol=3, columnspacing=0.5)

g6_inset = {}

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
        
        for j, file in enumerate(npz_files):
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


            ############ Part added for the inset ############
            if j == 0 and N == 80 and (density == 0.071 or density == 0.069):
                if f"n={density:.3f}" not in g6_inset:
                    g6_inset[f"n={density:.3f}"] = {}
                g6_inset[f"n={density:.3f}"]['radius_axis'] = radius_axis
                g6_inset[f"n={density:.3f}"]['estimators'] = estimators
            ##################################################


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

    marker = markers[i]
    colour = colours[i]
    light_colour = light_colours[i]
    axs[1].scatter(densities, integrated_g6s, marker=marker, color=light_colour, edgecolors=colour, linewidths=edgewidth, s=markersize, label=f'$N={N}$')

axs[1].axvspan(0.06811, 0.07159, alpha=0.2, color='gray', label='Obtained with the N=80 data')
# axs[1].axvspan(0.0680, 0.0711, alpha=0.2, color='gray',)# label='Liquid-solid coexistence\n[Gordillo & Ceperley 1998]')
axs[1].set_xlabel('$n \ [\mathrm{\AA}^{-2}]$')
axs[1].set_ylabel('$G_6$')  
axs[1].set_yscale('log')





################################################################################
#         Bottom insets: Plot g6(r) in the liquid and solid phases             #
################################################################################

ax_insets = [ax_inset_bottom_left, ax_inset_bottom_right]
inset_densities = [0.069,0.071]
for i, ax_inset in enumerate(ax_insets):
    radius_axis = np.real(g6_inset[f"n={inset_densities[i]:.3f}"]["radius_axis"])
    g6s = np.real(g6_inset[f"n={inset_densities[i]:.3f}"]["estimators"])
    ax_inset.plot(rm * radius_axis, g6s, color=colours[-1])
    ax_inset.plot(rm * radius_axis, g6s, color=colours[-1])
    ax_inset.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax_inset.set_xlabel('$r \ [\mathrm{\AA}]$', fontsize=6, labelpad=0.6)
    ax_inset.set_ylabel('$g_6(r)$', fontsize=6, labelpad=0.6)
    ax_inset.tick_params(axis='both', which='major', labelsize=6, pad=0.5)
    ax_inset.tick_params(axis='both', which='minor', labelsize=6, pad=0.5)
    # ax_inset.set_yscale('log')




################################################################################
#            Top insets: Plot g(r) in the liquid and solid phases              #
################################################################################

path = f'pair_correlation_function_data/N={N}/'

def extract_density(filename):
    return float(re.search(r'density=(\d+\.\d+)', filename).group(1))
filenames = os.listdir(path)
filenames = [f for f in filenames if f.startswith('pair_correlation_function') and f'N={N}' in f and f.endswith('.npz')]
filenames = [f for f in filenames if ('density=0.069' in f or 'density=0.071' in f)]
sorted_filenames = sorted(filenames, key=extract_density)
    
ax_insets = [ax_inset_top_left, ax_inset_top_right]
for i, ax_inset in enumerate(ax_insets):
    filename_head = sorted_filenames[i]
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
    y -= 1  # g2(x=0,y) - 1

    ax_inset.plot(x, y, color=colours[-1])
    ax_inset.plot(x, y, color=colours[-1])
    ax_inset.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax_inset.set_xlabel('$x \ [\mathrm{\AA}]$', fontsize=6, labelpad=0.6)
    ax_inset.set_ylabel('$g(x,0)-1$', fontsize=6, labelpad=0.6)
    ax_inset.tick_params(axis='both', which='major', labelsize=6, pad=0.5)
    ax_inset.tick_params(axis='both', which='minor', labelsize=6, pad=0.5)
    # ax_inset.set_yscale('log')

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
    ax_inset.plot(x_fit, fitted_y, color=red, linestyle='--', label='Fitted Decay Function')
    # ax_inset.scatter(x_local_maxima, y_local_maxima, color='red', marker='*', s=10, zorder=10) # , label='First Local Maximum'







plt.tight_layout()

## Reduce the spacing between axs[0] and axs[1] (should be done after `plt.tight_layout()`)
plt.subplots_adjust(hspace=0.05)

output_filename = 'peaks_structure_factor_and_integrated_g6_he4_vs_N=30-80_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()
