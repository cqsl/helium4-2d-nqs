import netket as nk
from netket.utils import mpi
import numpy as np
import jax.numpy as jnp
import jax
import flax 
from optax._src import linear_algebra
from trial_wavefunction import *
import optax
from typing import Callable
from netket.jax import tree_size
import shutil
from thermodynamic_potential import *
from utils import *


## Define physical and wavefunction variables
N = 31  # number of helium-4 atoms
d = 2  # spatial dimension
# density = 0.075  # [A^-d]
rm = 2.9673  # [A]
n_bf_transformations = 1
latent_dim = 6
n_features = 6
mlp_layers = 1
save_period = 100

# Mx = 4
# My = 2
# assert N == 2*Mx*My

# a = np.sqrt(2/(np.sqrt(3)*density))  # lattice constant, obtained by fixing rho=N/(Lx*Ly) [A]

# # lx x ly defines a rectangle that contains two unit cells (in terms of its area)
# lx = a  # [A]
# ly = np.sqrt(3)*a  # [A]
# Lx = Mx*lx  # [A]
# Ly = My*ly  # [A]

pressure = 2.4  # [KA^{-2}]
L_init = (21.,20.) # (Lx,Ly) # (3*a,3*a) 
theta_init = 1.4  # [2*jnp.pi/3, 1.834, jnp.pi/2, 1.45, 4*jnp.pi/10, jnp.pi/3]  #jnp.pi/2-0.1
geometry_init = L_init + (theta_init,)  # (Lx,Ly,theta)
L_init_str = '-'.join([f'{i:.2f}' for i in L_init])


path_to_scratch = '/home/linteau/scratch/'  # TODO: change this to your scratch path
output_filename = f'he4_N={N}_d={d}_p={pressure}_bt={n_bf_transformations}_ld={latent_dim}_nf={n_features}_hd={mlp_layers}_L-init={L_init_str}_theta-init={theta_init:.3f}'
mpack_filename = output_filename


# Define hyperparameters
sigma = 0.005
sweep_size = 32
n_chains_per_device = 64
n_samples_per_device = 1024
n_discard_per_chain = 32
chunk_size = 128
learning_rate = 0.001
diag_shift = 0.005

n_chains = n_chains_per_device * jax.device_count()
n_samples = n_samples_per_device * jax.device_count()

print(f'Simulation of {N} particles at a pressure of {pressure} KA^-2 in a rectangle of\n' 
f'initial size {L_init[0]:.4f} x {L_init[1]:.4f} A, with initial angle: {theta_init:.4f} rad.')
hilb = nk.hilbert.Particle(N=N, L=(1.,1.), pbc=True)  # sample a unit square box!!
sampler = nk.sampler.MetropolisGaussian(hilb, sigma=sigma, n_chains=n_chains, sweep_size=sweep_size)

pot_func = lambda x, p, Lx, Ly, theta: gibbs_potential(x, pressure=p, L=jnp.array((Lx,Ly)), theta=theta)
pot = nk.operator.PotentialEnergy(hilb, afun=pot_func, pressure=pressure, arguments=geometry_init)
ekin = nk.operator.KineticEnergy(hilb, arguments=geometry_init)
epsilon_tilde = 7.84637  # potential prefactor: epsilon*m_He*rm^2/hbar^2
ha = -0.5 * rm**2 * ekin + epsilon_tilde * pot

model = McMillanWithBackflowLxLyThetaParams(
        d=d,
        n_up=N,
        n_down=0,
        L=L_init,
        theta=theta_init,
        embedding_dim=latent_dim,
        intermediate_dim=latent_dim,
        mlp_layers=mlp_layers,
        attention_dim=latent_dim,
        n_features=n_features,
        n_interactions=n_bf_transformations,
        cusp_exponent=5,
) 
vs = nk.vqs.MCState(sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain, chunk_size=chunk_size)
print(f'There are {vs.n_parameters} parameters in the Lx-Ly free model.')


# # Load optimized parameters
# mpack_filename =  mpack_filename + '.mpack'
# with open(mpack_filename, 'rb') as file:
#     vs.variables = flax.serialization.from_bytes(vs.variables, file.read())


# ## Load samples to initialize the chains
# flag = '0'
# from netket.sampler import MetropolisSamplerState
# np.random.seed(eval(flag))
# n_nodes = nk.utils.mpi.n_nodes 
# key = jax.random.PRNGKey(eval(flag))
# keys = jax.random.split(key, num=n_nodes)
# samples_filename = sample_filename + '.npy'  # TODO: define a filename to load some samples
# samples = jnp.load(samples_filename)  # of the form (n_ranks, n_chains_per_rank, n_samples_per_rank, hilbert.size)
# d1, d2, d3, d4 = samples.shape
# x = samples[mpi.rank % d1,...]  # (d2, d3, d4)
# x = x.reshape(-1, d4)  # (d2 * d3, d4)
# replace = False if n_chains <= d2 * d3 else True
# rand_indices = np.random.choice(d2 * d3, size=n_chains, replace=replace)  # (n_chains,)
# x = x[rand_indices,:]  # (n_chains, hilbert.size)
# rng = keys[mpi.rank]
# st = MetropolisSamplerState(σ=x, rng=rng, rule_state=vs.sampler_state.rule_state)
# vs.sampler_state = st
# vs.reset()  # keep same σ as starting point (because `reset_chains=False` in the sampler) but take subkey of rng




## Define a special optimizer function that only optimizes a subset of parameters;
## Freezes all but one parameter (e.g. below 'w1') -- https://github.com/netket/netket/issues/916
def flattened_traversal(fn):
  """ Returns function that is called with `(path, param)` instead of pytree. 
  https://github.com/google/flax/discussions/1453"""
  def mask(tree):
    flat = flax.traverse_util.flatten_dict(tree)
    return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
  return mask
# label_fn = flattened_traversal(lambda path, _: True if (path[0] == 'Lx_param' or path[0] == 'Ly_param' or path[0] == 'theta_param') else False)
label_fn = flattened_traversal(lambda path, _: path[0] in ('Lx_param', 'Ly_param', 'theta_param'))
label_params_to_optimize = label_fn(vs.parameters)



## The S-matrix is regularized with a diagonal shift \espilon, via S_ij -> S_ij + \delta_ij \epsilon_j.
## Here since S_ij = 0 for all entries (i,j) associated to a geometric parameter, we put a diagonal shift of 1
## for these geometric parameters to recover a standard optimization procedure (e.g. SGD, Adam, etc.) -- in particular not SR.
## It's implicit here that the geometric parameters should be the first parameters defined in the wavefunction, since they
## should be associated with the unit diagonal shift entries (placed at the beginning of the diag_shift vector).
n_params = vs.n_parameters # tree_size(vs.parameters)
n_geometry_vars = len(geometry_init)
# diag_shift = jnp.array([1.,] * n_geometry_vars + [diag_shift,] * (n_params-n_geometry_vars))  # this line doesn't work anymore
diag_shift = jnp.concatenate([jnp.ones(n_geometry_vars), jnp.full(n_params - n_geometry_vars, diag_shift)])
diag_shift = flax.core.unfreeze(
    flax.traverse_util.unflatten_dict(
        {k: diag_shift[i] for i, k in enumerate(flax.traverse_util.flatten_dict(vs.parameters))}
    )
)

## RUN VMC!
op = optax.multi_transform({True: optax.adabelief(learning_rate=0.05), False: optax.sgd(learning_rate)}, label_params_to_optimize)  # optax.adabelief(learning_rate=0.005)  optax.set_to_zero()
# op = nk.optimizer.Sgd(learning_rate=learning_rate)
sr = nk.optimizer.SR(diag_shift=diag_shift)
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

def mycb(step, logged_data, driver):
    ## Store the sampling acceptance and the global norm of the gradient
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    logged_data["globalnorm"] = float(linear_algebra.global_norm(driver._loss_grad))

    Lx = vs.variables["params"]["Lx_param"][0]
    Ly = vs.variables["params"]["Ly_param"][0]
    theta = vs.variables["params"]["theta_param"][0]
    logged_data["Lx"] = Lx  # [A]
    logged_data["Ly"] = Ly  # [A]
    logged_data["theta"] = theta

    # ## Look for the k largest gradient entries in the gradient pytree
    # pytree = vs.grad(ha)
    # z = find_k_largest_values(pytree, k=4)
    # for x in z: print(x)

    # ## Periodically save samples and the model parameters
    # if step % save_period == 0 and step > 0:
    #     xs, _ = mpi.mpi_allgather_jax(vs.samples)
    #     jnp.save(sample_filename + f'_iter={step}.npy', xs)
    #     # mpack_filename_current = mpack_filename + '.mpack'
    #     # mpack_filename_copy = mpack_filename + f'_iter={step}.mpack'
    #     # shutil.copy(mpack_filename_current, mpack_filename_copy)

    return True

logger = nk.logging.JsonLog(
    output_prefix= output_filename,
    save_params_every=10,
    write_every=10,
)
gs.run(n_iter=30000,)# callback=mycb, out=logger)  # TODO: uncomment to store a log file and the parameter (mpack) file 
