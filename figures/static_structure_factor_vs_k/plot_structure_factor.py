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

def plot_setup_with_2_subplots(aspect_ratio=1/1.62, width_ratio=1., wide=False): 
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    fig, axs = plt.subplots(1, 2, figsize=(width,height/2), dpi=200, facecolor='white')
    return fig, axs

def plot_with_inset_setup(aspect_ratio=1/1.62, width_ratio=1., wide=False):
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width,height), dpi=200, facecolor='white')
    # these are in unitless percentages of the figure size. (0,0 is bottom left):
    left, bottom, width, height = [0.57, 0.57, 0.38, 0.35] 
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

blue = 'tab:blue'
light_blue = lighter(colors.to_rgb(blue), 0.7)








#############################################################################
#           S(|k|) vs |k| for two number densities (for insets)             #
#############################################################################

# rm = 2.9673  # [A]
# N = 30
# path = f'structure_factor_data/N={N}/'
# filenames = os.listdir(path)
# def extract_density(filename):
#     return float(re.search(r'density=([\d.]+)', filename).group(1))

# filenames = os.listdir(path)
# filenames = [filename for filename in filenames if filename.endswith('.npz') and ('density=0.065' in filename or 'density=0.075' in filename)]
# sorted_filenames = sorted(filenames, key=extract_density)

# fig, axs = plot_setup_with_2_subplots()
# densities = []

# for i, filename in enumerate(sorted_filenames):

#     phase = 'liquid' if 'liquid' in filename else 'solid'
#     N = int(re.search(r'N=(\d+)', filename).group(1))
#     density = extract_density(filename)
#     densities.append(density)

#     ## Load data
#     data = jnp.load(path + filename)
#     n_max = data['n_max']
#     ks = data['ks']/rm  # [A^-1]
#     estimators = data['estimators']

#     ## S(|k|) vs |k|
#     ks_norm = jnp.linalg.norm(ks, axis=-1)
#     mask = jnp.argsort(ks_norm)
#     ks_norm_sorted = ks_norm[mask]
#     ks_norm_unique, indices = jnp.unique(ks_norm_sorted, return_index=True)
#     struc_fac_splitted = jnp.split(jnp.real(estimators)[mask], indices_or_sections=indices)[1:]  
#     struc_fac_averaged = jnp.array([jnp.mean(x) for x in struc_fac_splitted])

#     # Avoid |k|=0
#     k = ks_norm_unique[1:]
#     s = struc_fac_averaged[1:]

#     ax = axs[i]
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))  # specify the number of y-ticks 
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))  
#     ax.tick_params(axis='x', labelsize=10)
#     ax.tick_params(axis='y', labelsize=10)
#     ax.scatter(k, s, color=light_blue, edgecolors=blue, linewidths=0.5, s=14)

#     ## Only put a y-label on the leftmost plot, and remove the x-labels (it will be put on the momentum distrubution plots)
#     if i == 0:
#         ax.set_ylabel('$S(k)$', fontsize=10)
#     ax.set_xticklabels([])

#     ax.set_xlim(-0.2,4.)  # set the same x-axis limit as for the momentum distribution (inset) plot

# output_filename = f'structure_factor_vs_k_N=30_he4_d=2_density={densities[0]:.3f}_{densities[1]:.3f}_bt=4_ld=8_nf=8_hd=1.pdf'
# plt.tight_layout()
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()











#############################################################################
#             Panel with density plots of many number densities             #
#############################################################################


# rm = 2.9673  # [A]
# N = 30
# path = f'structure_factor_data/N={N}/'
# filenames = os.listdir(path)
# def extract_density(filename):
#     return float(re.search(r'density=([\d.]+)', filename).group(1))

# filenames = os.listdir(path)
# filenames = [filename for filename in filenames if filename.endswith('.npz')]
# # filenames = [f for f in filenames if 'density=0.085' in f or 'density=0.090' in f]
# sorted_filenames = sorted(filenames, key=extract_density)

# fig, axs = plt.subplots(1, len(sorted_filenames), figsize=(25,5))

# for i, filename in enumerate(sorted_filenames):

#     phase = 'liquid' if 'liquid' in filename else 'solid'
#     N = int(re.search(r'N=(\d+)', filename).group(1))
#     density = extract_density(filename)

#     ## Load data
#     data = jnp.load(path + filename)
#     n_max = data['n_max']
#     ks = data['ks']/rm  # [A^-1]
#     estimators = data['estimators']

#     # 2D ONLY: Density plot S(kx,ky) vs kx, ky
#     k_min = jnp.min(ks)  # [A^-1]    
#     k_max = jnp.max(ks)  # [A^-1]    
#     ks_norm = jnp.linalg.norm(ks, axis=-1)
#     # k_min = jnp.min(ks_norm)  # [A^-1]
#     # k_max = jnp.max(ks_norm)  # [A^-1]
#     estimators = jnp.real(estimators)
#     estimators = estimators.at[jnp.argmax(estimators)].set(0.)  # smooth out the gamma point k=(0,0)

#     if density == 0.085 or density == 0.090:

#         ## Define physical and wavefunction variables
#         d = 2  # spatial dimension
#         Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
#         Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}
#         Mx = Mxs[f'N={N}']
#         My = Mys[f'N={N}']
#         a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
#         lx = a  # [A]
#         ly = np.sqrt(3)*a  # [A]
#         Lx = Mx*lx  # [A]
#         Ly = My*ly  # [A]
#         L = np.array([Lx,Ly]) / rm  # [dimensionless]
#         # print(f'Lx/rm = {Lx/rm}')
#         # print(f'Ly/rm = {Ly/rm}')
        
#         ## Define the direct space basis vectors
#         a1 = a * jnp.array([1.,0.])
#         a2 = a * jnp.array([1/2,np.sqrt(3)/2]) 
#         direct_basis = jnp.column_stack([a1,a2])  # [A]

#         ## Obtain the associated reciprocal space basis vectors
#         reciprocal_basis = 2 * jnp.pi * jnp.linalg.inv(direct_basis)  # [A^-1]
#         b1, b2 = reciprocal_basis
#         # print(f'b1 = {b1}')
#         # print(f'b2 = {b2}')

#         # ## Check the orthogonality constraints
#         # print(f'a1.b1/(2pi) = {jnp.dot(a1,b1)/(2*jnp.pi)}')  
#         # print(f'a2.b2/(2pi) = {jnp.dot(a2,b2)/(2*jnp.pi)}')  
#         # print(f'a1.b2/(2pi) = {jnp.dot(a1,b2)/(2*jnp.pi)}')  
#         # print(f'a2.b1/(2pi) = {jnp.dot(a2,b1)/(2*jnp.pi)}')  
        

#         Gs = ks[np.argsort(estimators)[-6:]]  # find the six largest values (i.e. the Bragg peaks)
#         print(f'Gs = {Gs}')

#         # axs[i].scatter(*reciprocal_basis.T, color='green', marker='D', s=50, zorder=10) 
#         # axs[i].scatter(*Gs.T, color='red', marker='o', s=30, zorder=10) 
#         # axs[i].scatter(*ks.T, color='orange', marker='.', s=5, zorder=10)

#     # max_sk = np.max(estimators)
#     axs[i].imshow(estimators.reshape(2*n_max+1, 2*n_max+1), extent=[k_min,k_max,k_min,k_max], origin='lower', cmap='viridis', vmin=0,)# vmax=max_sk)
#     axs[i].set_title(f'$n={density}$' + '$\ \mathrm{\AA}^{-2}$')
#     axs[i].axis('off')

# output_filename = f'panel_plot_structure_factor_vs_kx-ky_vs_density_N={N}_he4_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=400, bbox_inches='tight')
# plt.tight_layout()
# plt.show()
















#############################################################################
#                              Plot ks and Gs                               #
#############################################################################


rm = 2.9673  # [A]
N = 30
path = f'structure_factor_data/N={N}/'
filenames = os.listdir(path)
def extract_density(filename):
    return float(re.search(r'density=([\d.]+)', filename).group(1))

filenames = os.listdir(path)
filenames = [filename for filename in filenames if filename.endswith('.npz')]
# filenames = [f for f in filenames if 'density=0.085' in f or 'density=0.090' in f]
sorted_filenames = sorted(filenames, key=extract_density)

# fig, axs = plt.subplots(1, len(sorted_filenames), figsize=(25,5))

def get_reciprocal_lattice_vectors(direct_basis: jnp.array, n_max=1) -> jnp.array:
    """ The `direct_basis` argument should have shape (d,d) and contains
    the direct space basis vectors column-wise. """
    d = direct_basis.shape[0]
    reciprocal_basis = 2 * jnp.pi * jnp.linalg.inv(direct_basis)  # (d,d) [inverse length units]
    # b1, b2 = reciprocal_basis  # 2D only!
    n_step = 1  
    n_1d_axis = jnp.arange(-n_max, n_max + n_step, n_step)
    n_axes = jnp.tile(n_1d_axis[None], reps=(d,1))
    ns_grid = jnp.meshgrid(*n_axes)
    ns = jnp.vstack(ns_grid).reshape(d,-1).T  # ((2*n_max+1)**d, d)
    # Gs = ns[:,0,None] * b1[None,:]  + ns[:,1,None] * b2[None,:]  # 2D only! ((2*n_max+1)**d, d)
    Gs = jnp.dot(ns, reciprocal_basis)  # ((2*n_max+1)**d, d)
    return Gs


for i, filename in enumerate(sorted_filenames):

    if i == 0:

        phase = 'liquid' if 'liquid' in filename else 'solid'
        N = int(re.search(r'N=(\d+)', filename).group(1))
        density = extract_density(filename)

        ## Load data
        data = jnp.load(path + filename)
        n_max = data['n_max']
        ks = data['ks']/rm  # [A^-1]
        estimators = data['estimators']

        ## Define physical and wavefunction variables
        d = 2  # spatial dimension
        Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
        Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}
        Mx = Mxs[f'N={N}']
        My = Mys[f'N={N}']
        a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
        lx = a  # [A]
        ly = np.sqrt(3)*a  # [A]
        Lx = Mx*lx  # [A]
        Ly = My*ly  # [A]
        L = np.array([Lx,Ly]) / rm  # [dimensionless]
        print(L)
        # print(f'Lx/rm = {Lx/rm}')
        # print(f'Ly/rm = {Ly/rm}')
        
        ## Define the direct space basis vectors
        a1 = a * jnp.array([1.,0.])
        a2 = a * jnp.array([1/2,np.sqrt(3)/2]) 
        direct_basis = jnp.column_stack([a1,a2])  # [A]

        ## Obtain the associated reciprocal space basis vectors
        reciprocal_basis = 2 * jnp.pi * jnp.linalg.inv(direct_basis)  # [A^-1]
        b1, b2 = reciprocal_basis
        # print(f'b1 = {b1}')
        # print(f'b2 = {b2}')


        Gs = get_reciprocal_lattice_vectors(direct_basis, n_max=1)

        plt.scatter(*ks.T)
        plt.scatter(*Gs.T)

plt.show()






#############################################################################
#                                  S(|k-G|)                                 #
#############################################################################


# rm = 2.9673  # [A]
# N = 30
# path = f'structure_factor_data/N={N}_s_of_k_minus_G/'
# filenames = os.listdir(path)
# def extract_density(filename):
#     return float(re.search(r'density=([\d.]+)', filename).group(1))

# filenames = os.listdir(path)
# filenames = [filename for filename in filenames if filename.endswith('.npz') if 'density=0.090' in filename]
# sorted_filenames = sorted(filenames, key=extract_density)


# for i, filename in enumerate(sorted_filenames):

#     phase = 'liquid' if 'liquid' in filename else 'solid'
#     N = int(re.search(r'N=(\d+)', filename).group(1))
#     density = extract_density(filename)

#     ## Load data
#     data = jnp.load(path + filename)
#     n_max = data['n_max']
#     ks = data['ks']/rm  # [A^-1]
#     estimators = data['estimators']
#     s_of_k_minus_G = data['s_of_k_minus_G']

#     # 2D ONLY: Density plot S(kx,ky) vs kx, ky
#     k_min = jnp.min(ks)  # [A^-1]    
#     k_max = jnp.max(ks)  # [A^-1]    
#     ks_norm = jnp.linalg.norm(ks, axis=-1)
#     estimators = jnp.real(estimators)
#     estimators = estimators.at[jnp.argmax(estimators)].set(0.)  # smooth out the gamma point k=(0,0)

#     Gs = ks[np.argsort(estimators)[-7:-1]]  # find the six largest values (i.e. the Bragg peaks)
#     print(Gs)
#     Gs0 = Gs[2]
#     print(f'Gs0 = {Gs0}')
#     ks_minus_G = ks - Gs0
#     ks_minus_G_norm = jnp.linalg.norm(ks_minus_G, axis=-1)

#     print(ks[jnp.argmax(jnp.real(s_of_k_minus_G))])

#     plt.scatter(ks_norm, s_of_k_minus_G)

#     # plt.imshow(estimators.reshape(2*n_max+1, 2*n_max+1), extent=[k_min,k_max,k_min,k_max], origin='lower', cmap='viridis', vmin=0,)# vmax=max_sk)


# output_filename = f'panel_plot_structure_factor_vs_kx-ky_vs_density_N={N}_he4_d=2_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=400, bbox_inches='tight')
# plt.tight_layout()
# plt.show()
































# def rectangular_grid_arange(box_size: jnp.array, step_sizes: jnp.array) -> jnp.array:
#     d = box_size.shape[0]
#     assert step_sizes.size == d
#     coord_axes = [jnp.arange(box_size[i][0], box_size[i][1], step_sizes[i]) for i in range(d)]  # list of d sub-arrays of shape (num_coords[i],)
#     # coord_axes = [jnp.linspace(0, box_size[i], num_coords[i]) for i in range(d)]  # list of d sub-arrays of shape (num_coords[i],)
#     coord_grid = jnp.meshgrid(*coord_axes, indexing='ij')  # tuple with d sub-arrays of shape num_coords=(num_coords[0],...,num_coords[d-1])
#     grid_points = jnp.vstack([grid.flatten() for grid in coord_grid]).T  # (num_coords[0]*...*num_coords[d-1],d)
#     grid_spacings = []
#     for axis in coord_axes: 
#         grid_spacings.append(axis[1]-axis[0])
#     grid_spacings = jnp.array(grid_spacings)
#     return grid_spacings, grid_points


# def triangular_lattice(Mx, My, lx, ly):
#     """ To generate a triangular/hexagonal lattice for 2D helium-4 at solid densities. """
#     d = 2
#     Lx = Mx * lx
#     Ly = My * ly
#     L = jnp.array([Lx, Ly])
#     box_size = jnp.column_stack((jnp.zeros(d), L))
#     step_sizes = jnp.array([lx, ly])

#     _, grid_points = rectangular_grid_arange(box_size=box_size, step_sizes=step_sizes)
#     translation_vector = jnp.array([lx,ly]) / 2
#     lattice_vectors = jnp.concatenate((grid_points, grid_points + translation_vector))
    
#     ## To move the lattice points away from the boundary of the simulation cell
#     small_shift = jnp.array([lx,ly]) / 4 
#     # lattice_vectors += small_shift

#     return lattice_vectors 


# import itertools
# def reciprocal_vectors(direct_lattice_basis: jnp.array, n_max: int, num_q_vec: int) -> jnp.array: 
#     d = direct_lattice_basis.shape[0]
#     reciprocal_lattice_basis = 2 * jnp.pi * jnp.linalg.inv(direct_lattice_basis)  # (d,d)
#     integers = list(range(-n_max, n_max + 1))
#     combinations = jnp.array(list(itertools.product(integers, repeat=d)))
#     qs = combinations @ reciprocal_lattice_basis.T
#     # qs = combinations @ direct_lattice_basis.T
#     return qs[:num_q_vec]

# def create_cubic_grid(d: int, n_max=1) -> jnp.array:
#     """ Cubic d-dimensional grid of unit length symmetric w.r.t. (0,...,0). """
#     n_step = 1  
#     n_1d_axis = jnp.arange(-n_max, n_max + n_step, n_step)
#     n_axes = jnp.tile(n_1d_axis[None], reps=(d,1))
#     ns_grid = jnp.meshgrid(*n_axes)
#     ns = jnp.vstack(ns_grid).reshape(d,-1).T  # ((2*n_max+1)**d, d)
#     ns = ns[jnp.argsort(jnp.linalg.norm(ns, axis=-1))]  # sort by distance from the origin
#     return ns

# def smallest_wavevectors(L: float, d: int, n_max=15, num_k_vec=None) -> jnp.array:
#     """ Get the `num_k_vec` smallest (in 2-norm) reciprocal lattice vectors. """
#     ns = create_cubic_grid(d=d, n_max=n_max)
#     ks = 2 * jnp.pi * ns / L  # 1/[L]
#     if num_k_vec == None:
#         return ks
#     else:
#         mask = jnp.argsort(jnp.linalg.norm(ks, axis=-1))
#         ks = ks[mask]
#         return ks[:num_k_vec]  # take `num_k_vec` wavevectors with the smallest norm



# ## Define physical and wavefunction variables
# rm = 2.9673  # [A]
# N = 30  # number of helium-4 atoms
# d = 2  # spatial dimension
# density = 0.07  # [A^-d]
# Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
# Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}
# Mx = Mxs[f'N={N}']
# My = Mys[f'N={N}']
# a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]
# lx = a  # [A]
# ly = np.sqrt(3)*a  # [A]
# Lx = Mx*lx  # [A]
# Ly = My*ly  # [A]
# L = np.array([Lx,Ly]) / rm  # [dimensionless]
# print(f'Lx/rm = {Lx/rm}')
# print(f'Ly/rm = {Ly/rm}')


# a1 = a * jnp.array([1.,0.]) / rm
# a2 = a * jnp.array([1/2,np.sqrt(3)/2]) / rm
# direct_lattice_basis = jnp.column_stack([a1,a2])
# print(direct_lattice_basis)

# # qs = reciprocal_vectors(direct_lattice_basis, n_max=10, num_q_vec=3000)
# # mask = (0 <= qs[:, 0]) & (qs[:, 0] <= Lx/rm) & (0 <= qs[:, 1]) & (qs[:, 1] <= Ly/rm)
# # qs = qs[mask][:N]

# # Rs = triangular_lattice(Mx, My, lx/rm, ly/rm)
# # mask = (0 <= Rs[:, 0]) & (Rs[:, 0] <= Lx/rm) & (0 <= Rs[:, 1]) & (Rs[:, 1] <= Ly/rm)
# # Rs = Rs[mask][:N]

# # plt.scatter(*qs.T, marker='D')
# # plt.scatter(*Rs.T)

# # plt.axhline(0, color='black')
# # plt.axvline(0, color='black')
# # plt.axhline(Ly/rm, color='black')
# # plt.axvline(Lx/rm, color='black')

# # plt.show()

# ks = smallest_wavevectors(L=L, d=d, n_max=2, num_k_vec=None)
# print(ks)
# qs = reciprocal_vectors(direct_lattice_basis, n_max=10, num_q_vec=3000)

# plt.scatter(*qs.T, marker='D')
# plt.scatter(*ks.T)
# plt.show()