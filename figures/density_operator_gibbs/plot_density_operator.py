import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import re


# ## CANONICAL ENSEMBLE 
# current_dir = os.getcwd()
# filename_head = 'density_operator_he4_N=80_d=2_density=0.0705_bt=4_ld=8_nf=8_hd=1_branch=solid.npz'

# ## Load data
# data = jnp.load(os.path.join(current_dir, filename_head))
# num_coords = data['num_coords']
# Lx = data['Lx']
# Ly = data['Ly']
# estimators = data['estimators']

# plt.imshow(estimators.reshape(num_coords), extent=[0,Lx,0,Ly], origin='lower')

# plt.xlabel('$x \ [\AA]$')
# plt.ylabel('$y \ [\AA]$')
# # plt.axis('off')  # remove the axes and the associated labels

# # output_filename = current_dir + data_path + f'density_operator_he4_N=30_d=2_p={pressure}_bt=4_ld=6_nf=6_hd=1_phase={phase}.pdf'
# # plt.savefig(output_filename, dpi=400, bbox_inches='tight')
# plt.show()





## ISOBARIC ENSEMBLE

current_dir = os.getcwd()
data_path = '/isobaric_N=30/'

for f in os.listdir(current_dir + data_path):
    print(f)


def extract_pressure(filename):
    # Extract the pressure value from the filename
    return float(re.search('p=([\d\.]+)', filename).group(1))
filenames = os.listdir(current_dir + data_path)
filenames = [filename for filename in filenames if filename.endswith('.npz')]
sorted_filenames = sorted(filenames, key=extract_pressure)


for filename_head in sorted_filenames:

    ## Load data
    data = jnp.load(current_dir + data_path + filename_head)
    num_coords = data['num_coords']
    Lx = data['Lx']
    Ly = data['Ly']
    estimators = data['estimators']

    plt.imshow(estimators.reshape(num_coords), extent=[0,Lx,0,Ly], origin='lower', cmap='viridis')

    plt.xlabel('$x \ [\AA]$')
    plt.ylabel('$y \ [\AA]$')
    plt.axis('off')  # remove the axes and the associated labels

    pressure = extract_pressure(filename_head)
    phase = 'liquid' if 'liquid' in filename_head else 'solid'
    # plt.title(f'Phase: {phase}, Pressure = {pressure} KA^-2')
    
    output_filename = current_dir + data_path + f'density_operator_he4_N=30_d=2_p={pressure}_bt=4_ld=6_nf=6_hd=1_phase={phase}.pdf'
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
