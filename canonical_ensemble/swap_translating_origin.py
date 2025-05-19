#####################################################################################################################
#     TODO: this file is a bit of mess, and should be cleaned up... (the content is there, the form is lacking)     #
#####################################################################################################################

# Calculate the SWAP estimator using a spherical spatial bipartition of continuous space samples,
# either in 2D or 3D. The trial wavefunction optimized parameters are loaded
# and also the VMC samples. We use MPIs to speed up the process.

import netket as nk
import numpy as np
import jax.numpy as jnp
from trial_wavefunction import *
from netket.utils import mpi
import flax
import pickle 
from netket.utils.types import Array, PRNGKeyT
import jax
import os, re, sys, glob
import netket_extensions as nkext
from netket import jax as nkjax


# flag = str(eval(sys.argv[1])+0)  # to launch an array of jobs
flag = '1'  # to launch a single job
# flags = [str(i) for i in range(2,11,1)]
# for flag in flags:

rm = 2.9673  # [A]
pool_size = '130k' # N=80: 20*72*32~46k, N=30: 14*72*128~130k
ns = '50M' # '50M' # number of Monte Carlo samples in one bootstrap sample
# R_target = 11./rm  # [dimensionless]

## For free fermions testing
sigma = 0.05
n_sweeps = 1
n_chains_per_rank = 128
n_samples_per_rank = 128
n_discard_per_chain = 1024

# Ansatz variables
bt = 4
ld = 8
nf = 8
mlp_layers = 1

## Define physical and wavefunction variables
N = 30  # number of helium-4 atoms
d = 2  # spatial dimension

Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}

Mx = Mxs[f'N={N}']
My = Mys[f'N={N}']
assert N == 2*Mx*My

scratchpath = '/scratch/linteau/helium/'


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
        'he4_N=30_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072',
        'he4_N=30_d=2_density=0.072_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.071', 
        'he4_N=30_d=2_density=0.073_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072', 
        'he4_N=30_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073', 
        'he4_N=30_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid', 
        'he4_N=30_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid', 
        'he4_N=30_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid', 
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
        # 'he4_N=80_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.080',
        # 'he4_N=80_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.077',
        # 'he4_N=80_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.090',
        # 'he4_N=80_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid',
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
        'mc_samples_he4_N=30_d=2_density=0.071_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.072_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.071_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.073_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.072_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.073_iter=*.npy', 
        'mc_samples_he4_N=30_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
        'mc_samples_he4_N=30_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy', 
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
        # 'mc_samples_he4_N=80_d=2_density=0.075_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.080_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.080_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.077_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.085_bt=4_ld=8_nf=8_hd=1_branch=solid_init=0.090_iter=*.npy',
        # 'mc_samples_he4_N=80_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid_iter=*.npy',
    ]
}



num_files = len(filename_heads[f'N={N}'])
for i in range(num_files):
    
    filename_head = filename_heads[f'N={N}'][i]
    print(filename_head)
    mpack_filename = scratchpath + 'log/' + filename_head + ".mpack"
    samples_in_filename = scratchpath + 'samples/' + samples_filenames[f'N={N}'][i]
    samples_in_filename = glob.glob(samples_in_filename)[0]  # assuming only 1 filename with the pattern provided
    if mpi.rank == 0:
        print(samples_in_filename)
    density = float(re.search(r'density=([\d.]+)', filename_head).group(1))

    a = jnp.sqrt(2 / (jnp.sqrt(3) * density))  # lattice constant, obtained by fixing rho=N/(Lx*Ly) [A]
    # lx x ly defines a rectangle that contains two unit cells (in terms of its area)
    lx = a  # [A]
    ly = jnp.sqrt(3) * a  # [A]
    Lx = Mx * lx  # [A]
    Ly = My * ly  # [A]
    L = jnp.array([Lx,Ly]) / rm  # [dimensionless]

    samples_filename = scratchpath + 'samples/out_mc_samples_' + filename_head + f'_{pool_size}_v1.npy'
    # out_dir = scratchpath + f'estimators/num_R=10/' + filename_head + f'_{ns}_v{flag}/'
    observable_path = f'bootstrapping/N={N}/num_R=1/Rn12=2.39/changing_origins/'
    observable_path_raw = f'bootstrapping/N={N}/num_R=1/Rn12=2.39/raw/'
    out_dir = scratchpath + observable_path_raw + filename_head + f'_{ns}_v{flag}/'
    # out_dir = scratchpath + observable_path_raw + filename_head + f'_R-target={R_target:.2f}_{ns}_v{flag}/'

    ## Define the ansatz/trial wavefunction
    model = McMillanWithBackflow(
            d=d,
            n_up=N,
            n_down=0,
            L=tuple(L.tolist()),
            embedding_dim=ld,
            intermediate_dim=ld,
            mlp_layers=mlp_layers,
            attention_dim=ld,
            n_features=nf,
            n_interactions=bt,
            cusp_exponent=5,
    )

    ## Initialize the variational state interface to load and store the optimized parameters
    hilb = nk.hilbert.Particle(N=N, L=(Lx/rm,Ly/rm), pbc=True)
    # sab = nk.sampler.MetropolisGaussian(hilb, sigma=sigma, n_chains_per_rank=n_chains_per_rank, n_sweeps=n_sweeps)
    sab = nkext.sampler.MetropolisGaussAdaptive(hilb, initial_sigma=0.05, target_acceptance=0.5, n_chains_per_rank=n_chains_per_rank, n_sweeps=n_sweeps)
    vs = nk.vqs.MCState(sab, model, n_samples_per_rank=n_samples_per_rank, n_discard_per_chain=n_discard_per_chain)

    ## Load the optimized (ground state) parameters
    with open(mpack_filename, 'rb') as file:
        vs.variables = flax.serialization.from_bytes(vs.variables, file.read())
    logpsi = vs._apply_fun
    gs_params = vs.variables  # ground state optimized parameters


    def bootstrapping(init_sample_size: int, bootstrap_size: int, key: PRNGKeyT) -> Array:
        """ Random sampling of a pool of samples with replacement. The sampling is implicitly performed by generating
        random indices corresponding to the first axis of the pool of samples. We will split the computation
        using MPIs so we also need to make sure that there is an even number of samples for a given rank. """
        n_samples_per_rank = bootstrap_size // nk.utils.mpi.n_nodes  
        n_samples_per_rank = n_samples_per_rank if n_samples_per_rank % 2 == 0 else n_samples_per_rank-1  # <~~ can't be jitted
        bootstrap_size = n_samples_per_rank * nk.utils.mpi.n_nodes
        bootstrap_indices = jax.random.choice(key, init_sample_size, shape=(bootstrap_size,), replace=True)
        return n_samples_per_rank, bootstrap_indices
        
    def remove_pairs_of_identical_samples(indices):
        n_pairs_before_delete = indices.size // 2
        idx_pairs = indices.reshape(2,-1)
        idx_to_del = np.where(idx_pairs[0] == idx_pairs[1])[0] 
        indices_to_delete = np.concatenate((idx_to_del, idx_to_del + n_pairs_before_delete))
        indices = np.delete(indices, indices_to_delete)
        return indices

    @jax.jit
    def min_imag_conv_distance(x: Array, L: Array, origin: Array) -> Array:
        """ Compute distances between particles and the provided `origin` using the minimum image convention. 
        Takes `x` of shape (N,d), `L` and `origin` of shape (d,). """
        assert x.shape[-1] == L.size == origin.size
        mic_coords = jnp.remainder(x - origin + L/2.0, L) - L/2.0  # (N,d)
        return jnp.linalg.norm(mic_coords, axis=-1)  # (N,)


    @jax.jit
    def dist_from_orgin(R: Array, L: Array) -> Array:
        """ `R` has shape (M,N,d) and `L` has shape (d,). The origin is at the center of the box. """
        return jnp.linalg.norm(R - L/2, axis=-1)  # (M,N)

    @jax.jit
    def dist_from_orgins(R: Array, L: Array, origins: Array) -> Array:
        """ `R` has shape (M,N,d), `L` has shape (d,) and `origins` has shape (M,d). """
        return jax.vmap(min_imag_conv_distance, in_axes=(0, None, 0))(R, L, origins)  # (M,N)

    @jax.jit
    def sort_samples(R: Array, dist: Array) -> Array:
        """ `R` has shape (M,N,d) and `dist` has shape (M,N). This function
        sorts the 2nd dimension of `R`, corresponding to the particle index, according to the distance
        of each particle from the origin."""
        return R[jnp.arange(R.shape[0])[:,None], jnp.argsort(dist, axis=-1)]  # (M,N,d)

    @jax.jit
    def n_particles_in_partition_A(dist: Array, radius: float) -> Array:
        """ `dist` has shape (M,N). """
        num_particles_in_A = jnp.count_nonzero(dist < radius, axis=-1)  # (M,)
        return num_particles_in_A

    @jax.jit
    def swap_estimator(x1: Array, x2: Array, x3: Array, x4: Array) -> float:
        """ `xi`'s have shape (N*d,). """
        logpsi_x1, logpsi_x2, logpsi_x3, logpsi_x4 = logpsi(gs_params, jnp.vstack((x1,x2,x3,x4)))
        swap = jnp.exp(logpsi_x1 + logpsi_x2 - logpsi_x3 - logpsi_x4)
        return swap

    @jax.jit
    def swap_non_trivial_contribution(R1: Array, R2: Array, N1A: Array, N2A: Array) -> float:
        """ 'Ri''s have shape (N*d,) and are *importantly sorted*, and 'NiA''s have shape (1,). """
        idx = jnp.arange(R1.shape[-1])  # (N*d,)
        R1A_R2B = jnp.where(idx < N1A*d, R1, 0.) 
        R1A_R2B += jnp.where(idx < N2A*d, 0., R2)
        R2A_R1B = jnp.where(idx < N2A*d, R2, 0.) 
        R2A_R1B += jnp.where(idx < N1A*d, 0., R1)
        swap = swap_estimator(R1A_R2B, R2A_R1B, R1, R2)
        return swap

    @jax.jit
    def swap_non_trivial_contribution_diff_origins(R1: Array, R2: Array, N1A: Array, N2A: Array, t1: Array, t2: Array) -> float:
        """ 'Ri''s have shape (N*d,) and are *importantly sorted*, and 'NiA''s have shape (1,). """
        idx = jnp.arange(R1.shape[-1])  # (N*d,)
        assert t1.size == t2.size
        d = t1.size
        _R1 = (R1.reshape(-1,d) - t1 + t2).reshape(R1.shape)
        _R2 = (R2.reshape(-1,d) - t2 + t1).reshape(R2.shape)
        R1A_R2B = jnp.where(idx < N1A*d, _R1, 0) 
        R1A_R2B += jnp.where(idx < N2A*d, 0, R2)
        R2A_R1B = jnp.where(idx < N2A*d, _R2, 0) 
        R2A_R1B += jnp.where(idx < N1A*d, 0, R1)
        swap = swap_estimator(R1A_R2B, R2A_R1B, R1, R2)
        return swap

    @jax.jit
    def swap_wrapped(R1: Array, R2: Array, N1A: Array, N2A: Array) -> float:
        """ 'Ri''s have shape (N*d,) and 'NiA''s have shape (1,). """
        swap = jax.lax.cond(
            N1A == N2A,
            lambda x: swap_non_trivial_contribution(*x),
            lambda x: 0., # he4: 0., he3: 0.+0.j,
            (R1, R2, N1A, N2A),
        )
        return swap

    @jax.jit
    def swap_wrapped_diff_origins(R1: Array, R2: Array, N1A: Array, N2A: Array, t1: Array, t2: Array) -> float:
        """ 'Ri''s have shape (N*d,) and 'NiA''s have shape (1,). """
        swap = jax.lax.cond(
            N1A == N2A,
            lambda x: swap_non_trivial_contribution_diff_origins(*x),
            lambda x: 0., # he4: 0., he3: 0.+0.j,
            (R1, R2, N1A, N2A, t1, t2),
        )
        return swap
        
    @jax.jit
    def counter_nz_cont(N1A: Array, N2A: Array) -> float:
        return jax.lax.cond(N1A == N2A, lambda : 1., lambda : 0.)

    @jax.jit
    def wrapped_all(xs: Array, box_size: Array, radius: float) -> Array:
        """ Return all SWAP values obtained from the pairs of samples in this one bootstrap sample. """
        n_samples, N, d = xs.shape  
        n_pairs = n_samples//2  

        key = jax.random.PRNGKey(eval(flag))
        key = nkjax.mpi_split(key)
        # dist = dist_from_orgin(xs, box_size)  # distance from the origin of each single-particle coordinate (n_samples,N)
        origins = jax.random.uniform(key, shape=(n_samples, d), minval=jnp.array([0.,]*d), maxval=box_size)  # (n_samples,d)
        dist = dist_from_orgins(xs, box_size, origins)  # distance from the origin of each single-particle coordinate (n_samples,N)

        xs = sort_samples(xs, dist)  # sort the single-particle coordinates based on `dist` (n_samples,N,d)
        xs = xs.reshape(n_samples,N*d)
        na = n_particles_in_partition_A(dist, radius)  # (n_samples,)
        
        # swaps = jax.vmap(swap_wrapped, in_axes=(0,0,0,0))(xs[:n_pairs], xs[n_pairs:], na[:n_pairs], na[n_pairs:])
        # scan_fun = lambda carry, x: (carry, swap_wrapped(*x))  # if `vmap` too memory intensive use `scan`
        # _, swaps = jax.lax.scan(scan_fun, None, (xs[:n_pairs], xs[n_pairs:], na[:n_pairs], na[n_pairs:]), length=n_pairs)
        scan_fun = lambda carry, x: (carry, swap_wrapped_diff_origins(*x))  # if `vmap` too memory intensive use `scan`
        _, swaps = jax.lax.scan(scan_fun, None, (xs[:n_pairs], xs[n_pairs:], na[:n_pairs], na[n_pairs:], origins[:n_pairs], origins[n_pairs:]), length=n_pairs)
        scan_fun = lambda carry, x: (carry, counter_nz_cont(*x))  # to get the swap acceptance rate
        _, nz_cont = jax.lax.scan(scan_fun, None, (na[:n_pairs], na[n_pairs:]), length=n_pairs)
        # swaps, _ = mpi.mpi_allgather_jax(swaps)  # (nk.utils.mpi.n_nodes, n_pairs)
        return nz_cont, swaps.ravel()   # (nk.utils.mpi.n_nodes * n_pairs,)


    ## Initialize the sampler state by loading samples
    from netket.sampler import MetropolisSamplerState
    np.random.seed(eval(flag))
    n_nodes = nk.utils.mpi.n_nodes 
    key = jax.random.PRNGKey(eval(flag))
    keys = jax.random.split(key, num=n_nodes)
    samples = jnp.load(samples_in_filename)  # of the form (n_ranks, n_chains_per_rank, n_samples_per_rank, hilbert.size)
    d1, d2, d3, d4 = samples.shape
    x = samples[mpi.rank % d1,...]  # (d2, d3, d4)
    x = x.reshape(-1, d4)  # (d2 * d3, d4)
    replace = False if n_chains_per_rank <= d2 * d3 else True
    rand_indices = np.random.choice(d2 * d3, size=n_chains_per_rank, replace=replace)  # (n_chains_per_rank,)
    x = x[rand_indices,:]  # (n_chains_per_rank, hilbert.size)
    rng = keys[mpi.rank]
    st = MetropolisSamplerState(σ=x, rng=rng, rule_state=vs.sampler_state.rule_state)
    vs.sampler_state = st
    vs.reset()  # keep same σ as starting point (because `reset_chains=False` in the sampler) but take subkey of rng





    ################################################################################
    #                    Save VMC samples to compute the entropy                   #
    ################################################################################

    # xs, _ = mpi.mpi_allgather_jax(vs.samples)
    # print(xs.shape)
    # if mpi.rank == 0:
    #     jnp.save(samples_filename, xs)





    ################################################################################
    #    Aggregate data (store only the mean or put all estimators in 1 file)      #
    ################################################################################


    if mpi.rank == 0:  # do the following only on one rank

        ## Unpack files from each mpi process: only store the mean of all the (truncated) SWAP estimators for each R:
        def truncated_swap(estimators, tolerance_on_1=1e-12, method='truncated1'):
            """ Given many instances of the swap estimator, return a truncated version of it. """
            tolerance_on_1 = 1e-12 
            mask_equal_to_1 = np.abs(estimators-1.) < tolerance_on_1
            print(f'Number of estimators equal to 1 (for each R): {np.sum(mask_equal_to_1, axis=-1)}')
            mask_strictly_less_than_1 = np.abs(np.real(estimators)) < 1.-tolerance_on_1    ######################### |Re[.]| < 1
            if method == 'truncated1':
                trunc_estimators = np.where(mask_strictly_less_than_1, 2*np.real(estimators)+1j*0, 0.+0.j)  ######################### np.real
                trunc_estimators += np.where(mask_equal_to_1, estimators, 0.+0.j)
            elif method == 'truncated2':
                trunc_estimators = np.where(mask_equal_to_1 | mask_strictly_less_than_1, estimators, 0.)
                # trunc_estimators += np.where(mask_strictly_less_than_1, estimators, 0.)
            else:
                raise ValueError(f"Method {method} not implemented.")
            return trunc_estimators

        ## No need to re-run the code with these lines:
        from time import sleep
        while True:
            n_files = len(os.listdir(out_dir))
            if n_files == nk.utils.mpi.n_nodes or nk.utils.mpi.n_nodes == 1:   
                for flag in range(1,6,1):
                # for flag in [eval(flag),]:
                    flag = str(flag)
                    scratchpath = '/scratch/linteau/helium/'
                    out_dir = scratchpath + observable_path_raw + filename_head + f'_{ns}_v{flag}/'
                    input_dir = out_dir
                    estimators = []
                    nz_contributions = []
                    file_count = 0
                    for input_filename in os.listdir(input_dir):
                        file_count += 1
                        with open(input_dir + input_filename, 'rb') as f:
                            data = pickle.load(f)
                        R = data['R_axis']
                        estimators.append(np.array(data['tr_rho_a2']))
                        nz_contributions.append(np.array(data['nz_contributions']))
                    print(f'Number of files processed: {file_count}')
                    estimators = np.hstack(estimators)  # (num_R, n_files * num_estimators_per_file)
                    # swap_mean = np.mean(jnp.abs(estimators), axis=-1)  # (num_R,)
                    nz_contributions = np.hstack(nz_contributions)  # (num_R, n_files * num_estimators_per_file)
                    print(f'Swap acceptance rate vs R: {jnp.sum(nz_contributions, axis=-1) / nz_contributions.shape[-1]}')
                    truncated_estimators = truncated_swap(estimators)
                    estimators = np.mean(truncated_estimators, axis=-1)  # (num_R,)
                    print(f'Swap mean vs R: {estimators}')

                    ## Save data (version 1)
                    # res = {"R_axis": R, "tr_rho_a2": swap_mean.tolist()}
                    # print(res)
                    # print(f'S2={-np.log(swap_mean)}')
                    # # Save to a file
                    # out_path = scratchpath + observable_path + in_filename + f'_truncated-and-averaged_{ns}_v{flag}.pkl'
                    # with open(out_path, "wb") as f:
                    #     pickle.dump(res, f)

                    # ## Save data (version 2)
                    # # estimators = estimators[:,:1000]
                    # # out_path = scratchpath + f'estimators/num_R=1/' + in_filename + f'_R={R_target*rm:.1f}A_{ns}_v{flag}.npz' 
                    # density_path = f'density={density:.3f}/'
                    # out_path = scratchpath + observable_path + density_path + in_filename + f'_{ns}_v{flag}.npz' 
                    # np.savez(out_path, R_axis=R, estimators=estimators, nz_contributions=nz_contributions)

                    ## Save data (version 3)
                    branch = 'liquid' if 'liquid' in filename_head else 'solid'
                    density_path = f'density={density:.3f}_branch={branch}/'
                    # density_path = f'density={density:.3f}/'
                    # density_path = f'density={density:.3f}_R-target={R_target*rm:.2f}A/'
                    full_path = scratchpath + observable_path + density_path
                    # Check if the directory exists and create it if it doesn't
                    if not os.path.exists(full_path):
                        os.makedirs(full_path)
                    out_path = full_path + filename_head + f'_{ns}_v{flag}.npz'
                    print(f'output path: {out_path}')
                    np.savez(out_path, R_axis=R, estimators=estimators, nz_contributions=nz_contributions)

                break  # stop the while loop!
            
            sleep(10)  # wait for 10 seconds before checking again
