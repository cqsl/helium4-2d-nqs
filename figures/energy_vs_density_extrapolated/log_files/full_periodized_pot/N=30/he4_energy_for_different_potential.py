# import os
# os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
import jax.numpy as jnp
import numpy as np
import jax
import netket as nk
from trial_wavefunction_new import *
from optax._src import linear_algebra
import flax
from netket.utils import mpi
import netket_extensions as nkext
import json
import re, os
import glob
import scipy.integrate as integrate
from jax.scipy.special import gamma
from jax.scipy.integrate import trapezoid


def create_grid(d: int, n_max=1) -> jnp.array:
    """ Cubic d-dimensional grid of unit length. """
    n_step = 1  
    n_1d_axis = jnp.arange(-n_max, n_max + n_step, n_step)
    n_axes = jnp.tile(n_1d_axis[None], reps=(d,1))
    ns_grid = jnp.meshgrid(*n_axes)
    ns = jnp.vstack(ns_grid).reshape(d,-1).T  # ((2*n_max+1)**d, d)
    ns = ns[jnp.argsort(jnp.linalg.norm(ns, axis=-1))]  # sort by distance from the origin
    return ns

def coordinate_diff(x: jnp.array) -> jnp.array:
    """ Compute the coordinate differences rij = ri - rj. """
    N = x.shape[0]
    return (x[jnp.newaxis, :, :] - x[:, jnp.newaxis, :])[jnp.triu_indices(N, 1)]

def min_imag_conv_distance(x: jnp.array, L: jnp.array) -> jnp.array:
    """Compute distances between particles using minimum image convention"""
    rij = coordinate_diff(x)
    rij_mic = jnp.remainder(rij + L / 2.0, L) - L / 2.0  # (N(N-1)/2,d)
    return jnp.linalg.norm(rij_mic, axis=-1)

def aziz(dis: jnp.array) -> jnp.array:
    eps = 7.846373
    A = 0.544850 * 10**6
    alpha = 13.353384
    c6 = 1.3732412
    c8 = 0.4253785
    c10 = 0.178100
    D = 1.241314 
    return eps * (
        A * jnp.exp(-alpha * dis)
        - (c6 / dis**6 + c8 / dis**8 + c10 / dis**10)
        * jnp.where(dis < D, jnp.exp(-((D / dis - 1) ** 2)), 1.0)
    )

## To compute the volume of a d-dimensional sphere of radius r
vol_d_r = lambda d, r:  jnp.pi**(d/2) * r**d / gamma(d/2+1)  

def v_shift_fun(N: int, density: float, d: int, rc: float) -> float:
    """ Shift contribution to the potential. Note that [density]=[rc]=dimensionless. """
    Nc = vol_d_r(d, rc) * density  # number of particles in the sphere of radius rc
    v_shift = 0.5 * N * Nc * aziz(rc) 
    return v_shift  # [dimensionless]

def v_tail_fun(N: int, density: float, d: int, rc: float) -> float:
    """ Tail contribution to the potential. Note that [density]=[rc]=dimensionless. """
    # v_tail = N * density * jnp.pi**(d/2) * integrate.quad(lambda r: r**(d-1) * aziz(r), rc, jnp.inf)[0] / gamma(d/2)
    integrand_fun = lambda r: r**(d-1) * aziz(r)
    r_axis = jnp.linspace(rc, 10*rc, 1000)  # take a higher bound sufficiently large to approximate infinity
    v_tail = N * density * jnp.pi**(d/2) * trapezoid(y=integrand_fun(r_axis), x=r_axis) / gamma(d/2)
    return v_tail  # [dimensionless]

def potential(x: jnp.array, L: jnp.array) -> float:
    """ x \in \vec{L}/rm [dimensionless] and [`L`]=dimensionless """
    rm = 2.9673  # [A]
    d = L.size  # spatial dimension
    x = x.reshape(-1, d)  # (N,d)
    N = x.shape[0]

    # ## Potential over the nearest image (using either the minimum image convention distance or the sine distance)
    # Lmin = jnp.min(L) 
    # rc = Lmin/2  
    # v_rc = aziz(rc) 
    # distances = min_imag_conv_distance(x, L)  # (N(N-1)/2,)
    # v = jnp.sum(jnp.where(distances <= rc, aziz(distances) - v_rc, 0.))  
    
    # ## Constant potential contributions (importantly they depend on the box size)
    # density = N / jnp.prod(L) 
    # v += v_shift_fun(N, density, d, rc)
    # v += v_tail_fun(N, density, d, rc)

    ## Full periodized potential (over nearest neighbours (NN), next NN, up to some cutoff)
    rij = coordinate_diff(x)  # (N(N-1)/2,d) 
    ns = create_grid(d=d, n_max=2)  # (X,d)=((2*n_max+1)**d,d)
    n_times_L = ns * L  # (X,d) 
    rij_minus_n_times_L = rij[:, None, :] - n_times_L[None, :, :]  # (N(N-1)/2, X, d) 
    distances = jnp.linalg.norm(rij_minus_n_times_L, axis=-1).flatten()  # (N(N-1)/2 * X,)
    v = jnp.sum(aziz(distances)) # periodic images
    
    ## Constant potential contributions (importantly they depend on the box size);
    ## If sufficiently many images are considered, the shift and tail contributions should be negligible.
    madelung_distances = jnp.linalg.norm(n_times_L[1:], axis=-1)  # don't sum over n=(0,0), i.e. `ns[0]`
    v_madelung = 0.5 * jnp.sum(aziz(madelung_distances))
    v += N * v_madelung

    return v  # [dimensionless]





flag = '1'

n_gpus = jax.device_count()

## Define physical and wavefunction variables
N = 80  # number of helium-4 atoms
d = 2  # spatial dimension
potential_type = 'full-periodized' # 'truncated-shifted'

Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}

Mx = Mxs[f'N={N}']
My = Mys[f'N={N}']
assert N == 2*Mx*My

filename_heads = {
    "N=30": [
        'he4_N=30_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1_branch=liquid',  
        'he4_N=30_d=2_density=0.055_bt=4_ld=8_nf=8_hd=1_branch=liquid',  
        'he4_N=30_d=2_density=0.060_bt=4_ld=8_nf=8_hd=1_branch=liquid',  
        'he4_N=30_d=2_density=0.065_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.066',  
        'he4_N=30_d=2_density=0.066_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.067',  
        'he4_N=30_d=2_density=0.067_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.068',  
        'he4_N=30_d=2_density=0.068_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.069',  
        'he4_N=30_d=2_density=0.069_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.070',  
        'he4_N=30_d=2_density=0.070_bt=4_ld=8_nf=8_hd=1_branch=liquid',
        ######### # 'he4_N=30_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid', 
        'he4_N=30_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072',
        'he4_N=30_d=2_density=0.072_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.071', 
        'he4_N=30_d=2_density=0.073_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072', 
        'he4_N=30_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073', 
        'he4_N=30_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid', 
        'he4_N=30_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid', 
        'he4_N=30_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid', 
    ],
    "N=56": [
        # 'he4_N=56_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1_branch=liquid',
        # 'he4_N=56_d=2_density=0.055_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.050',
        # 'he4_N=56_d=2_density=0.060_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.055',
        'he4_N=56_d=2_density=0.065_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.060',
        'he4_N=56_d=2_density=0.066_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.065',
        'he4_N=56_d=2_density=0.067_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.066',
        'he4_N=56_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.085',
        'he4_N=56_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.090',
        # 'he4_N=56_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid',
    ],
    "N=80": [
        'he4_N=80_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1_branch=liquid',  
        'he4_N=80_d=2_density=0.055_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.050',
        'he4_N=80_d=2_density=0.060_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.065',
        'he4_N=80_d=2_density=0.065_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.066', 
        'he4_N=80_d=2_density=0.066_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.067', 
        'he4_N=80_d=2_density=0.067_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.068', 
        'he4_N=80_d=2_density=0.068_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.0685',
        # 'he4_N=80_d=2_density=0.0685_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.068',
        'he4_N=80_d=2_density=0.069_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.0685',
        # 'he4_N=80_d=2_density=0.0695_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.069',
        'he4_N=80_d=2_density=0.070_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.0695',
        'he4_N=80_d=2_density=0.070_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.0705',
        # 'he4_N=80_d=2_density=0.0705_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.071',
        # 'he4_N=80_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072',
        'he4_N=80_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072_v2',
        'he4_N=80_d=2_density=0.072_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073',
        'he4_N=80_d=2_density=0.073_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.075',
        # 'he4_N=80_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073',
        'he4_N=80_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.080',
        'he4_N=80_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.077',
        'he4_N=80_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.090',
        'he4_N=80_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid',
    ]
}



samples_filenames = {
    "N=30": [
        'mc_samples_he4_N=30_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1_branch=liquid_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.055_bt=4_ld=8_nf=8_hd=1_branch=liquid_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.060_bt=4_ld=8_nf=8_hd=1_branch=liquid_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.065_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.066_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.066_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.067_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.067_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.068_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.068_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.069_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.069_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.070_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.070_bt=4_ld=8_nf=8_hd=1_branch=liquid_iter=*.npy', 
        ###### # 'mc_samples_he4_N=30_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.072_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.071_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.073_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy', 
        ],
    "N=56": [
        # 'mc_samples_he4_N=56_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1_branch=liquid_iter=*.npy',
        # 'mc_samples_he4_N=56_d=2_density=0.055_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.050_iter=*.npy',
        # 'mc_samples_he4_N=56_d=2_density=0.060_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.055_iter=*.npy',
        'mc_samples_he4_N=56_d=2_density=0.065_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.060_iter=*.npy',
        'mc_samples_he4_N=56_d=2_density=0.066_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.065_iter=*.npy',
        'mc_samples_he4_N=56_d=2_density=0.067_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.066_iter=*.npy',
        'mc_samples_he4_N=56_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.085_iter=*.npy',
        'mc_samples_he4_N=56_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.090_iter=*.npy',
        # 'mc_samples_he4_N=56_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
    ],
    "N=80": [
        'mc_samples_he4_N=80_d=2_density=0.050_bt=4_ld=8_nf=8_hd=1_branch=liquid_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.055_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.050_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.060_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.065_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.065_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.066_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.066_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.067_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.067_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.068_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.068_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.0685_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.0685_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.068_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.069_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.0685_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.0695_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.069_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.070_bt=4_ld=8_nf=8_hd=1_branch=liquid_init=0.0695_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.070_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.0705_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.0705_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.071_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072_iter=*.npy',  
        'mc_samples_he4_N=80_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072_v2_iter=*.npy',  
        'mc_samples_he4_N=80_d=2_density=0.072_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073_iter=*.npy',  
        'mc_samples_he4_N=80_d=2_density=0.073_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.075_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.080_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.077_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.090_iter=*.npy',
        'mc_samples_he4_N=80_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
    ]
}



for i in range(len(filename_heads[f'N={N}'])):  # [A^-d]
    
    filename_head = filename_heads[f'N={N}'][i]
    mpack_filename = 'scratch/data_log/' + filename_head + ".mpack"
    output_filename = 'scratch/data_log_diff_pot/' + filename_head + "_" + potential_type
    samples_filename ='scratch/samples/' + samples_filenames[f'N={N}'][i]
    samples_filename = glob.glob(samples_filename)[0]  # assuming only 1 filename with the pattern provided
    density = float(re.search(r'density=([\d.]+)', filename_head).group(1))

    rm = 2.9673  # [A]
    bt = 4  # number of backflow transformations
    ld = 8  # latent dimension
    nf = 8  # number of features
    hl = 1  # number of MLP hidden layers

    # ## 3D
    # L = (N/density)**(1/d)/rm
    # L = jnp.array((L,)*d)  # [dimensionless]

    ## 2D
    a = np.sqrt(2 / (np.sqrt(3) * density))  # lattice constant, obtained by fixing rho=N/(Lx*Ly) [A]
    # lx x ly defines a rectangle that contains two unit cells (in terms of its area)
    lx = a  # [A]
    ly = np.sqrt(3) * a  # [A]
    Lx = Mx * lx  # [A]
    Ly = My * ly  # [A]
    L = jnp.array([Lx,Ly]) / rm  # [dimensionless]
    # L = tuple(L.tolist()) 

    ## Instantiate the model
    sigma = 0.05  
    n_sweeps = 32
    n_chains_per_rank = 4096
    n_samples_per_rank = 4096 
    n_discard_per_chain = 32
    hilb = nk.hilbert.Particle(N=N, L=(Lx/rm,Ly/rm), pbc=True)
    # sampler = nk.sampler.MetropolisGaussian(hilb, sigma=0.01, n_chains_per_rank=n_chains_per_rank, n_sweeps=n_sweeps) ## [TO EVALUATE THE DENSITY OPERATOR!!!!]
    sampler = nkext.sampler.MetropolisGaussAdaptive(hilb, initial_sigma=sigma, target_acceptance=0.5, n_chains_per_rank=n_chains_per_rank, n_sweeps=n_sweeps)
    pot = nk.operator.PotentialEnergy(hilb, lambda x: potential(x, L))
    ekin = nkext.operator.KineticEnergy(hilb, mass=1.0, mode='folx')
    ha = ekin + pot
    model = McMillanMLPwithMPNN(
        d=d,
        n_up=N,
        n_down=0,
        L=tuple(L.tolist()),
        embedding_dim=ld,
        intermediate_dim=ld,
        mlp_layers=hl,
        attention_dim=ld,
        n_features=nf,
        n_interactions=bt,
        cusp_exponent=5,
    )
    vs = nk.vqs.MCState(
        sampler,
        model,
        n_samples_per_rank=n_samples_per_rank,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=4,
    )
    print(f"There are {vs.n_parameters} parameters in the model.")
    ## Load the ground state optimized parameters
    with open(mpack_filename, "rb") as file:
        vs.variables = flax.serialization.from_bytes(vs.variables, file.read())


    ## Initialize the sampler state by loading samples
    import numpy as np
    from netket.sampler import MetropolisSamplerState
    import jax
    np.random.seed(eval(flag))
    n_nodes = nk.utils.mpi.n_nodes 
    key = jax.random.PRNGKey(eval(flag))
    keys = jax.random.split(key, num=n_nodes)
    samples = jnp.load(samples_filename)  # of the form (n_ranks, n_chains_per_rank, n_samples_per_rank, hilbert.size)
    d1, d2, d3, d4 = samples.shape
    x = samples[mpi.rank % d1,...]  # (d2, d3, d4)
    x = x.reshape(-1, d4)  # (d2 * d3, d4)
    replace = False if n_chains_per_rank <= d2 * d3 else True
    rand_indices = np.random.choice(d2 * d3, size=n_gpus * n_chains_per_rank, replace=replace)  # (n_chains_per_rank,)
    x = x[rand_indices,:]  # (n_gpus * n_chains_per_rank, hilbert.size)
    rng = keys[mpi.rank]
    st = MetropolisSamplerState(σ=x, rng=rng, rule_state=vs.sampler_state.rule_state)
    vs.sampler_state = st
    vs.reset()  # keep same σ as starting point (because `reset_chains=False` in the sampler) but take subkey of rng


    ## RUN VMC!
    op = nk.optimizer.Sgd(learning_rate=0.)
    sr = nk.optimizer.SR(diag_shift=0., diag_scale=0.)
    gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

    def mycb(step, logged_data, driver):
        logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
        logged_data["globalnorm"] = float(linear_algebra.global_norm(driver._loss_grad))
        return True

    logger = nk.logging.JsonLog(
        output_prefix=output_filename,
        save_params_every=20,
        write_every=20,
    )

    gs.run(n_iter=1, callback=mycb, out=logger)