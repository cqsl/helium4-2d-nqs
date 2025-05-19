import json
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
import numpy as np
import os 
import math
from brokenaxes import brokenaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    # # these are in unitless percentages of the figure size. (0,0 is bottom left):
    # left, bottom, width, height = [0.64, 0.48, 0.3, 0.43] 
    # ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset = inset_axes(ax, width="60%", height="60%", loc='upper right')
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
red = convert_rgb_255_to_1_scale((186, 30, 15))

# light_blue = lighter(colors.to_rgb(blue), 0.7)
# light_green = lighter(colors.to_rgb(green), 0.7)

colours = [blue, green, red]
# light_colours = [light_blue, light_green]










## Define global variables
best_vmc_nqs_energy = {
    'N=30': {
        'n=0.075': 1.020,
    },
    'N=80': {
        'n=0.075': 1.085,
        'n=0.090': 4.312,
    },
}  # [K]
best_vmc_u3_energy = {
    'N=30': {
        'n=0.075': 1.101,
    },
    'N=80': {
        'n=0.075': 1.195,
    },
}  # [K]
part_of_optimization = '2'  # '1', '2' or '3'

ns = 1024  # number of (optimized) samples to consider 
alphaHe4 = 1.37643  
alphaHe3 = 1.66185  
alpha = alphaHe4  # factor to reintroduce units (of kelvin)


current_dir = os.getcwd()
folderpath = current_dir + '/data_log/' 




# #####################################################################
# #               Transfer learning (N=30, n=0.075 A^-2)              #
# #####################################################################

# N = 30
# d = 2
# density = 0.075 # 1/Ang^d
# phase = 'solid'

# bt = 4
# ld = 8
# nf = 8
# mlp_layers = 1

# datafilename = f'he4_N={N}_d=2_density={density:.3f}_bt={bt}_ld={ld}_nf={nf}_hd={mlp_layers}_branch=solid_p{part_of_optimization}.log'


# ## Unpack the data
# data_log = json.load(open(folderpath + datafilename))
# iters = np.array(data_log['Energy']['iters'])
# tot_energies = np.array(data_log['Energy']['Mean'])



# fig = plot_setup()
# gs = fig.add_gridspec(1, 1)

# # Create the broken axes plot
# time_segments = ((0,500),(2000,2500),(20000,20500))
# energy_segments =  ((0.98,1.1),(1.7,1.77))
# bax = brokenaxes(xlims=time_segments, ylims=energy_segments, hspace=0.5, wspace=0.5)

# bax.plot(iters, alpha * tot_energies / N)
# bax.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)

# # Add labels and legend
# bax.set_xlabel('Iterations')
# bax.set_ylabel('$E/N$ [K]')

# ## To artificially add top and right borders to the bax plot.
# ## It's important to add this subplot after the bax subplot to make 
# ## the borders appear on top of the brokenaxes plot.
# ax0 = fig.add_subplot(gs[0])
# ax0.set_xticks([])
# ax0.set_yticks([])
# ax0.spines['left'].set_visible(False)
# ax0.spines['bottom'].set_visible(False)
# ax0.patch.set_alpha(0)  # Make the interior of the plot transparent

# output_filename = f'transfer_learning_energy_vs_iters_he4_N=30_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()





# #####################################################################
# #                    Training (N=80, n=0.09 A^-2)                   #
# #####################################################################

# N = 80
# d = 2
# density = 0.09 # 1/Ang^d
# phase = 'solid'

# bt = 4
# ld = 8
# nf = 8
# mlp_layers = 1

# datafilename = f'he4_N={N}_d=2_density={density:.3f}_bt={bt}_ld={ld}_nf={nf}_hd={mlp_layers}_branch=solid_p{part_of_optimization}.log'


# ## Unpack the data
# data_log = json.load(open(folderpath + datafilename))
# iters = np.array(data_log['Energy']['iters'])
# tot_energies = np.array(data_log['Energy']['Mean'])



# fig = plot_setup()
# gs = fig.add_gridspec(1, 1)

# # Create the broken axes plot
# time_segments = ((0,700),(6600,7000))
# energy_segments = None # ((4.25,4.5),(5,6.2),(10.5,11.))
# bax = brokenaxes(xlims=time_segments, ylims=energy_segments, hspace=0.5, wspace=0.5)

# bax.plot(iters, alpha * tot_energies / N)
# bax.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)

# # Add labels and legend
# bax.set_xlabel('Iterations')
# bax.set_ylabel('$E/N$ [K]')

# ## To artificially add top and right borders to the bax plot.
# ## It's important to add this subplot after the bax subplot to make 
# ## the borders appear on top of the brokenaxes plot.
# ax1 = fig.add_subplot(gs[0])
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.spines['left'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.patch.set_alpha(0)  # Make the interior of the plot transparent

# # Create inset of the zoomed region
# # ax_inset = inset_axes(ax0, width="60%", height="60%", loc='upper right')
# ax_inset = inset_axes(ax1, width="60%", height="60%", 
#                       bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),  
#                       bbox_transform=ax1.transAxes,
#                       loc='upper right')
# ax_inset.plot(iters, alpha * tot_energies / N)
# ax_inset.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)
# ax_inset.set_xlim(6600, 7000)
# ax_inset.set_ylim(4.25, 4.4)

# output_filename = f'transfer_learning_energy_vs_iters_he4_N=30_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()




# #####################################################################
# #               Transfer learning (N=80, n=0.075 A^-2)              #
# #####################################################################

# N = 80
# d = 2
# density = 0.075 # 1/Ang^d
# phase = 'solid'

# bt = 4
# ld = 8
# nf = 8
# mlp_layers = 1

# datafilename = f'he4_N={N}_d=2_density={density:.3f}_bt={bt}_ld={ld}_nf={nf}_hd={mlp_layers}_branch=solid_p{part_of_optimization}.log'


# ## Unpack the data
# data_log = json.load(open(folderpath + datafilename))
# iters = np.array(data_log['Energy']['iters'])[1:]  # discard the first point which is off due to initial unthermalized sampling
# tot_energies = np.array(data_log['Energy']['Mean'])[1:]



# fig = plot_setup()
# gs = fig.add_gridspec(1, 1)

# # Create the broken axes plot
# time_segments = None #((0,700),(6600,7000))
# energy_segments = ((1.05,1.15),(1.4,1.56))
# bax = brokenaxes(xlims=time_segments, ylims=energy_segments, hspace=0.5, wspace=0.5)

# bax.plot(iters, alpha * tot_energies / N)
# bax.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)

# # Add labels and legend
# bax.set_xlabel('Iterations')
# bax.set_ylabel('$E/N$ [K]')

# ## To artificially add top and right borders to the bax plot.
# ## It's important to add this subplot after the bax subplot to make 
# ## the borders appear on top of the brokenaxes plot.
# ax1 = fig.add_subplot(gs[0])
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.spines['left'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.patch.set_alpha(0)  # Make the interior of the plot transparent


# output_filename = f'transfer_learning_energy_vs_iters_he4_N=30_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1.pdf'
# # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# plt.show()




























def plot_setup(aspect_ratio=1/1.62, width_ratio=1., wide=False): 
    width = (_wide_width if wide else _width) * width_ratio
    height = width * aspect_ratio
    width *= 3.
    return plt.figure(figsize=(width,height), dpi=200, facecolor='white')


fig = plot_setup()
gs = fig.add_gridspec(1, 3,)#width_ratios=[1.2,1.2,1.2])

#####################################################################
#               Transfer learning (N=30, n=0.075 A^-2)              #
#####################################################################

N = 30
d = 2
density = 0.075 # 1/Ang^d
phase = 'solid'

bt = 4
ld = 8
nf = 8
mlp_layers = 1

datafilename = f'he4_N={N}_d=2_density={density:.3f}_bt={bt}_ld={ld}_nf={nf}_hd={mlp_layers}_branch=solid_p{part_of_optimization}.log'

## Unpack the data
data_log = json.load(open(folderpath + datafilename))
iters = np.array(data_log['Energy']['iters'])
tot_energies = np.array(data_log['Energy']['Mean'])

fig = plot_setup()

# Create the broken axes plot
time_segments = ((0,500),(20000,20500))
energy_segments =  ((0.98,1.11),(1.7,1.77))
bax = brokenaxes(xlims=time_segments, ylims=energy_segments, hspace=0.5, wspace=0.7, subplot_spec=gs[0], d=0.005)

bax.plot(iters, alpha * tot_energies / N, color=colours[0])
bax.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)
bax.axhline(best_vmc_u3_energy[f'N={N}'][f'n={density:.3f}'], color=colours[2], linestyle=':', zorder=10)
bax.set_title('Transfer learning\n($N=30$, $n=0.075$ $\mathrm{\AA}^{-2}$)')

# Add labels and legend
bax.set_xlabel('Iterations')
bax.set_ylabel('$E/N$ [K]')

## To artificially add top and right borders to the bax plot.
## It's important to add this subplot after the bax subplot to make 
## the borders appear on top of the brokenaxes plot.
ax0 = fig.add_subplot(gs[0])
ax0.set_xticks([])
ax0.set_yticks([])
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.patch.set_alpha(0)  # Make the interior of the plot transparent


#####################################################################
#                    Training (N=80, n=0.09 A^-2)                   #
#####################################################################

N = 80
d = 2
density = 0.09 # 1/Ang^d
phase = 'solid'

bt = 4
ld = 8
nf = 8
mlp_layers = 1

datafilename = f'he4_N={N}_d=2_density={density:.3f}_bt={bt}_ld={ld}_nf={nf}_hd={mlp_layers}_branch=solid_p{part_of_optimization}.log'

## Unpack the data
data_log = json.load(open(folderpath + datafilename))
iters = np.array(data_log['Energy']['iters'])
tot_energies = np.array(data_log['Energy']['Mean'])

# Create the broken axes plot
time_segments = ((0,700),(6600,7000))
energy_segments = None # ((4.25,4.5),(5,6.2),(10.5,11.))
bax = brokenaxes(xlims=time_segments, ylims=energy_segments, hspace=0.5, wspace=0.5, subplot_spec=gs[1], d=0.005)

bax.plot(iters, alpha * tot_energies / N, color=colours[0])
bax.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)
bax.set_title('Training\n($N=80$, $n=0.090$ $\mathrm{\AA}^{-2}$)')

# Add labels and legend
bax.set_xlabel('Iterations')

## To artificially add top and right borders to the bax plot.
## It's important to add this subplot after the bax subplot to make 
## the borders appear on top of the brokenaxes plot.
ax1 = fig.add_subplot(gs[1])
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.patch.set_alpha(0)  # Make the interior of the plot transparent

# Create inset of the zoomed region
# ax_inset = inset_axes(ax0, width="60%", height="60%", loc='upper right')
ax_inset = inset_axes(ax1, width="60%", height="60%", 
                      bbox_to_anchor=(0.05, 0.05, 0.9, 0.9),  
                      bbox_transform=ax1.transAxes,
                      loc='upper right')
ax_inset.plot(iters, alpha * tot_energies / N, color=colours[0])
ax_inset.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)
ax_inset.set_xlim(6600, 7000)
ax_inset.set_ylim(4.25, 4.4)

ax1.text(0.4, 0.15, '2 hours', color=colours[1], transform=ax1.transAxes,)
ax1.axvline(x=0.45, ymin=0, ymax=0.13, color=colours[1], linestyle='solid')

ax1.text(0.72, 0.15, '24 hours', color=colours[1], transform=ax1.transAxes,)
ax1.axvline(x=0.95, ymin=0, ymax=0.13, color=colours[1], linestyle='solid')


#####################################################################
#               Transfer learning (N=80, n=0.075 A^-2)              #
#####################################################################

N = 80
d = 2
density = 0.075 # 1/Ang^d
phase = 'solid'

bt = 4
ld = 8
nf = 8
mlp_layers = 1

datafilename = f'he4_N={N}_d=2_density={density:.3f}_bt={bt}_ld={ld}_nf={nf}_hd={mlp_layers}_branch=solid_p{part_of_optimization}.log'

## Unpack the data
data_log = json.load(open(folderpath + datafilename))
iters = np.array(data_log['Energy']['iters'])[1:]  # discard the first point which is off due to initial unthermalized sampling
tot_energies = np.array(data_log['Energy']['Mean'])[1:]

# Create the broken axes plot
time_segments = None #((0,700),(6600,7000))
energy_segments = ((1.05,1.2),(1.4,1.56))
bax = brokenaxes(xlims=time_segments, ylims=energy_segments, hspace=0.5, wspace=0.5, subplot_spec=gs[2], d=0.005)

bax.plot(iters, alpha * tot_energies / N, color=colours[0])
bax.axhline(best_vmc_nqs_energy[f'N={N}'][f'n={density:.3f}'], color='k', linestyle='--', zorder=10)
bax.axhline(best_vmc_u3_energy[f'N={N}'][f'n={density:.3f}'], color=colours[2], linestyle=':', zorder=10)
bax.set_title('Transfer learning\n($N=80$, $n=0.075$ $\mathrm{\AA}^{-2}$)')

# Add labels and legend
bax.set_xlabel('Iterations')

## To artificially add top and right borders to the bax plot.
## It's important to add this subplot after the bax subplot to make 
## the borders appear on top of the brokenaxes plot.
ax1 = fig.add_subplot(gs[2])
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.patch.set_alpha(0)  # Make the interior of the plot transparent


output_filename = f'transfer_learning_energy_vs_iters_he4_N=30-80_d=2_density=0.075-0.090_bt=4_ld=8_nf=8_hd=1.pdf'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()