# ## Uncomment the 2 following lines to run on multi-GPUs
# import os
# os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'
import netket as nk
import jax.numpy as jnp
import flax.linen as nn
import jax
from optax._src import linear_algebra
from jax import lax
import flax
from netket.utils import mpi
import numpy as np
import random
from trial_wavefunction import *
import netket_extensions as nkext
from utils import * 
from hamiltonian import potential


## Define physical and wavefunction variables
N = 30  # number of helium-4 atoms
d = 2  # spatial dimension
density = 0.05  # [A^-d]
phase = 'liquid'
rm = 2.9673  # [A]
n_bf_transformations = 1
latent_dim = 8
n_features = 8
mlp_layers = 1
save_period = 100

Mxs = {'N=30': 5, 'N=56': 7, 'N=64': 8, 'N=80': 8}
Mys = {'N=30': 3, 'N=56': 4, 'N=64': 4, 'N=80': 5}

Mx = Mxs[f'N={N}']
My = Mys[f'N={N}']
assert N == 2*Mx*My

a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant -- obtained by fixing rho=N/(Lx*Ly) [A]

# lx x ly defines a rectangle that contains two unit cells (in terms of its area)
lx = a  # [A]
ly = np.sqrt(3)*a  # [A]
Lx = Mx*lx  # [A]
Ly = My*ly  # [A]
L = jnp.array([Lx,Ly]) / rm  # [dimensionless]

scratchpath = '/home/linteau/scratch/'  # TODO: change this to your scratch path
output_filename = f'he4_N={N}_d={d}_density={density:.3f}_bt={n_bf_transformations}_ld={latent_dim}_nf={n_features}_hd={mlp_layers}_branch={phase}'
mpack_filename = output_filename


# Define hyperparameters
# the "n_discard_per_chain" and "n_sweeps" variables can be lowered provided the particles are not initialized too
# close to each other (one could load samples or modify the "random_state" method of the sampling rule)
sigma = 0.05 
n_sweeps = 64
n_chains_per_rank = 16
n_chains = n_chains_per_rank * jax.device_count()
n_samples_per_rank = 512 
n_samples = n_samples_per_rank * jax.device_count()
n_discard_per_chain = 64
chunk_size = 8
learning_rate = 0.0005  
diag_shift = 0.01


print(f'Simulation of {N} particles in a rectangle of size {Lx/rm:.4f} x {Ly/rm:.4f}.')
hilb = nk.hilbert.Particle(N=N, L=(Lx/rm,Ly/rm), pbc=True)
# sampler = nk.sampler.MetropolisGaussian(hilb, sigma=sigma, n_chains_per_rank=n_chains_per_rank, n_sweeps=n_sweeps)
sampler = nkext.sampler.MetropolisGaussAdaptive(hilb, initial_sigma=sigma, target_acceptance=0.5, n_chains=n_chains, n_sweeps=n_sweeps)

pot = nk.operator.PotentialEnergy(hilb, lambda x: potential(x, L)) 
# ekin = nk.operator.KineticEnergy(hilb, mass=1.0)
ekin = nkext.operator.KineticEnergy(hilb, mass=1.0, mode='folx')  # Forward Laplacian implementation
ha = ekin + pot

model = McMillanWithBackflow(
        d=d,
        n_up=N,
        n_down=0,
        L=tuple(L.tolist()),
        embedding_dim=latent_dim,
        intermediate_dim=latent_dim,
        mlp_layers=mlp_layers,
        attention_dim=latent_dim,
        n_features=n_features,
        n_interactions=n_bf_transformations,
        cusp_exponent=5,  # could be left as a variational parameter (provided it's properly initialized)
) 
vs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain, chunk_size=chunk_size)
print(f'There are {vs.n_parameters} parameters in the model.')


# # ## Load optimized parameters
# # mpack_filename = "pre-saved_mpack/" + mpackfile + ".mpack"
# mpack_filename = scratchpath + 'log/' + mpack_filename + '.mpack'
# with open(mpack_filename, 'rb') as file:
#     vs.variables = flax.serialization.from_bytes(vs.variables, file.read())



# ## Load samples to initialize the chains
# flag = '0'
# from netket.sampler import MetropolisSamplerState
# np.random.seed(eval(flag))
# n_nodes = nk.utils.mpi.n_nodes 
# key = jax.random.PRNGKey(eval(flag))
# keys = jax.random.split(key, num=n_nodes)
# samples_filename = scratchpath + 'samples/mc_samples_he4_N=80_d=2_density=0.090_bt=4_ld=8_nf=8_hd=1_branch=solid_v2_v7200.npy'
# samples = jnp.load(samples_filename)  # of the form (n_ranks, n_chains_per_rank, n_samples_per_rank, hilbert.size)
# d1, d2, d3, d4 = samples.shape
# x = samples[mpi.rank % d1,...]  # (d2, d3, d4)
# x = x.reshape(-1, d4)  # (d2 * d3, d4)
# replace = False if n_chains <= d2 * d3 else True
# rand_indices = np.random.choice(d2 * d3, size=n_chains, replace=replace)  # (n_chains_per_rank,)
# x = x[rand_indices,:]  # (n_chains_per_rank, hilbert.size)
# rng = keys[mpi.rank]
# st = MetropolisSamplerState(σ=x, rng=rng, rule_state=vs.sampler_state.rule_state)
# vs.sampler_state = st
# vs.reset()  # keep same σ as starting point (because `reset_chains=False` in the sampler) but take subkey of rng


## Gather samples 
# xs, _ = mpi.mpi_allgather_jax(vs.samples)
# xs = xs.reshape(-1,N,d)
# jnp.save(scratchpath + 'samples/positions_' + output_filename + f'.npy', xs)
# print(xs.shape)

# def find_k_largest_values(pytree, k=1):
#     flat_tree = flax.traverse_util.flatten_dict(pytree)
#     max_values = {key: jnp.max(val) for key, val in flat_tree.items()}
#     largest_k_values = dict(sorted(max_values.items(), key=lambda x: x[1])[-k:])
#     return largest_k_values 


## RUN VMC!
op = nk.optimizer.Sgd(learning_rate=learning_rate)
sr = nk.optimizer.SR(diag_shift=diag_shift)
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

def mycb(step, logged_data, driver):
    ## Store the sampling acceptance and the global norm of the gradient
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    logged_data["globalnorm"] = float(linear_algebra.global_norm(driver._loss_grad))
    # logged_data["ekin"] = vs.expect(ekin)
    # logged_data["epot"] = vs.expect(pot)

    # pytree = vs.grad(ha)
    # z = find_k_largest_values(pytree, k=4)
    # for k, v in z.items(): print(k, v)

    # ## Periodically save samples 
    # if step % save_period == 0:
    #     xs, _ = mpi.mpi_allgather_jax(vs.samples)
    #     jnp.save(scratchpath + 'samples/mc_samples_' + output_filename + f'_v{step}.npy', xs)
    return True

# logger = nk.logging.JsonLog(
#     output_prefix=scratchpath + 'log/' + output_filename,
#     save_params_every=20,
#     write_every=20,
# )
gs.run(n_iter=30000,)# callback=mycb, out=logger)  # TODO: uncomment to store a log file and the parameter (mpack) file 
