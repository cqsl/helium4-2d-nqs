from netket.vqs import MCState, expect
from observables.operator import (
    DensityOperator,
    OneBodyDensityMatrix,
    StructureFactor,
    MomentumDistribution,
    HexaticOrientationalOrderParameter,
    HexaticOOPFourier,
    PairCorrelationFunction,
)
import jax.numpy as jnp
import jax
from netket.utils import mpi
from functools import partial
from typing import Optional 
from ..utils import mic_distance, coordinate_differences, sort_samples, _angle_btw_2_vectors
from netket.jax import vmap_chunked


## gibbs N=30: sigma=0.03; n_sweeps = 8, n_chains_per_rank = 1, n_samples_per_rank = 8192, n_discard_per_chain = 2048
@jax.jit
def _density_operator_expect(mc_samples: jnp.array, r: jnp.array, sigma: float=0.03) -> float:  # N=80: 0.2
    """ Compute \rho(r) = \sum_{i=1}^N \delta(r - r_i), where r is a vector in the simulation cell 
    and r_i is a single-particle coordinate (MC sample). We approximate the delta function with a
    peaked Gaussian distribution. Assume `mc_samples` has shape (M,N,d) and `r` has shape (d,). """
    estimators = jnp.exp(-0.5 * jnp.linalg.norm(r - mc_samples, axis=-1)**2 / sigma**2)  # (M,N)
    estimators /= (sigma * jnp.sqrt(2 * jnp.pi))
    estimators = jnp.sum(estimators, axis=-1)  # (M,)
    return jnp.mean(estimators)  # (M,)->1


@expect.dispatch
def density_operator_expect(
    vstate: MCState,
    op: DensityOperator,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    rs = op._grid_points  # (num_points[0]*...*num_points[d-1],d)
    d = rs.shape[-1]  # spatial dimension

    samples = vstate.samples  # (n_chains_per_rank,n_samples_per_chain,N*d)
    N = samples.shape[-1] // d  # number of particles
    samples = jnp.squeeze(samples).reshape(-1,N,d)  # (n_chains_per_rank*n_samples_per_chain,N,d)

    estimators = jax.vmap(_density_operator_expect, in_axes=(None, 0))(samples, rs)  # (num_points[0]*...*num_points[d-1],)
    estimators, _ = mpi.mpi_allgather_jax(estimators)  # (nk.utils.mpi.n_nodes, num_points[0]*...*num_points[d-1])
    estimators = jnp.mean(estimators, axis=0)  # (num_points[0]*...*num_points[d-1],)

    return estimators


@partial(jax.jit, static_argnames=("idx"))
def _displace_one_particle(
    x: jnp.array, displacement_vector: jnp.array, idx: int
) -> jnp.array:
    """ `x` corresponds to batches of MC samples being displaced and
    `idx` corresponds to the particle index in each batch. """
    d = displacement_vector.size  # spatial dimension
    N = x.shape[-1] // d  # x.shape=(M,N*d)
    shift = jnp.pad(displacement_vector, (d * idx, d * (N - 1 - idx)))  # (d,) -> (N*d,)
    ## No need to do `(x+shift) % box_size` because the wavefunction itself is periodic
    return x + shift  # (M,N*d)


@partial(jax.jit, static_argnames=("vs"))
def _one_body_density_matrix_estimator(
    vs: MCState, x: jnp.array, x_prime: jnp.array
) -> jnp.array:
    """ Compute \rho^{(1)}(r)=\Psi(r_1+r,...,r_N)/Psi(r_1,...,r_N), for some displacement vector r.
    `x_prime` is a shifted version of `x` in only 1 of the N single-particle coordinates. Both arrays 
    are batched arrays with shape (M,N*d)."""
    log_psi = vs._apply_fun
    gs_vars = vs.variables
    logpsi_x_prime = log_psi(gs_vars, x_prime)
    logpsi_x = log_psi(gs_vars, x)
    idx_max = jnp.argmax(jnp.exp(logpsi_x_prime - logpsi_x))
    return jnp.exp(logpsi_x_prime - logpsi_x)  # (M,)


@partial(jax.jit, static_argnames=("vs"))
def _obdm_expect(vs: MCState, mc_samples: jnp.array, r: jnp.array) -> float:
    """Shift the ith particle by a d-dimensional vector. x has shape (M,N*d)."""
    shifted_samples = _displace_one_particle(
        mc_samples, r, idx=1
    )  # w.l.o.g displace particle 1
    estimators = _one_body_density_matrix_estimator(vs, mc_samples, shifted_samples)
    
    ## Remove outliers values (that are too large)
    threshold = 1e2
    mask = ~(estimators > threshold)  
    num_not_outliers = jnp.sum(mask)
    estimators = jnp.where(mask, estimators, 0.)
    mean_estimator = jnp.sum(estimators) / num_not_outliers

    # mean_estimator = jnp.mean(estimators)  # (M,) -> 1
    return mean_estimator


@expect.dispatch
def obdm_expect(
    vstate: MCState,
    op: OneBodyDensityMatrix,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    rs = op._displacement_vectors  # (num_points[0]*...*num_points[d-1],d)

    samples = vstate.samples  # (n_chains_per_rank,n_samples_per_chain,N*d)
    samples = jnp.squeeze(samples).reshape(-1, samples.shape[-1])  # (n_chains_per_rank*n_samples_per_chain,N*d)

    # estimators = jax.vmap(_obdm_expect, in_axes=(None, None, 0))(vstate, samples, rs)  # (num_points[0]*...*num_points[d-1],)
    
    # estimators = vmap_chunked(_obdm_expect, in_axes=(None, None, 0), chunk_size=chunk_size)(vstate, samples, rs)

    scan_fun = lambda carry, r: (carry, _obdm_expect(vstate, samples, r))
    _, estimators = jax.lax.scan(scan_fun, None, rs, length=rs.shape[0])

    estimators, _ = mpi.mpi_allgather_jax(estimators)  # (nk.utils.mpi.n_nodes, num_points[0]*...*num_points[d-1])

    estimators = jnp.mean(estimators, axis=0)  # (num_points[0]*...*num_points[d-1],)
    return estimators


@jax.jit
def _structure_factor_expect(samples: jnp.array, L: jnp.array, k: jnp.array) -> float:
    """ Compute S(k)=\rho_k \rho_{-k}/N = \sum_{i,j} e^{i*k.(r_i-r_j)}/N.
    Take `samples` of shape (M,N,d) and both `L` and `k` of shape (d,). """
    N = samples.shape[1]
    ri_minus_rj = mic_distance(samples, L)  # (M,N,N,d)
    estimators = jnp.exp(1j * jnp.einsum('l,ijkl->ijk', k, ri_minus_rj))  # (M,N,N)
    estimators = jnp.sum(estimators, axis=(1,2))  # (M,)
    estimators = jnp.mean(estimators)  # average over the batches
    return estimators / N


@expect.dispatch
def structure_factor_expect(
    vstate: MCState,
    op: StructureFactor,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    ks = op._wavevectors  # (num_k_vec,d)
    d = ks.shape[-1]  # spatial dimension
    L = jnp.array(op.hilbert.extent)

    samples = vstate.samples  # (n_chains_per_rank,n_samples_per_chain,N*d)
    N = samples.shape[-1] // d  # number of particles
    samples = jnp.squeeze(samples).reshape(-1,N,d)  # (n_chains_per_rank*n_samples_per_chain,N,d)

    # estimators = jax.vmap(_structure_factor_expect, in_axes=(None, None, 0))(samples, L, ks)  # (num_k_vec,)
    estimators = vmap_chunked(_structure_factor_expect, in_axes=(None, None, 0), chunk_size=chunk_size)(samples, L, ks)
    estimators, _ = mpi.mpi_allgather_jax(estimators)  # (nk.utils.mpi.n_nodes, num_k_vec)
    estimators = jnp.mean(estimators, axis=0)  # (num_k_vec,)

    # ########################################################################
    # ## Compute S(|k-G|)
    # Gs = ks[jnp.argsort(estimators)[-7:-1]]  # find the six largest values (i.e. the Bragg peaks), avoiding the gamma point (6,d)
    # Gs0 = Gs[2]  # take one of them w.l.o.g (d,)  
    # ks -= Gs0  # (num_k_vec,d)
    # s_of_k_minus_G = vmap_chunked(_structure_factor_expect, in_axes=(None, None, 0), chunk_size=chunk_size)(samples, L, ks)
    # s_of_k_minus_G, _ = mpi.mpi_allgather_jax(s_of_k_minus_G)  # (nk.utils.mpi.n_nodes, num_k_vec)
    # s_of_k_minus_G = jnp.mean(s_of_k_minus_G, axis=0)  # (num_k_vec,)
    # ########################################################################

    return estimators # , s_of_k_minus_G


@jax.jit
def _momentum_distribution_expect(dr: jnp.array, rs: jnp.array, obdms: jnp.array, k: jnp.array) -> float:
    """ Compute n_k*V = N_k*V/N =  \int dr e^{i*k.r} \rho^{(1)}(r), where \rho^{(1)} is the one-body
    density matrix defined above. We assume `rs` has shape (X,d), `obdms` (X,) and `k`  (d,). """
    d = k.size  # spatial dimension
    integrand = jnp.exp(1j * jnp.einsum('j,ij->i', k, rs)) * obdms  # (X,)
    return jnp.prod(dr) * jnp.sum(integrand, axis=-1) 


@expect.dispatch
def momentum_distribution_expect(
    vstate: MCState,
    op: MomentumDistribution,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    dr = op._grid_spacings
    rs = op._grid_points  # (num_points[0]*...*num_points[d-1],d)
    ks = op._wavevectors  # (num_k_vec,d)
    d = ks.shape[-1]  # spatial dimension

    obdm_op = OneBodyDensityMatrix(hilbert=op.hilbert, displacement_vectors=rs)
    obdm_estimators = vstate.expect(obdm_op)  # (num_points[0]*...*num_points[d-1],)

    # estimators = jax.vmap(_momentum_distribution_expect, in_axes=(None, None, None, 0))(dr, rs, obdm_estimators, ks)  # (num_k_vec,)
    
    # estimators = vmap_chunked(_momentum_distribution_expect, in_axes=(None, None, None, 0), chunk_size=chunk_size)(dr, rs, obdm_estimators, ks)
    
    scan_fun = lambda carry, k: (carry, _momentum_distribution_expect(dr, rs, obdm_estimators, k))
    _, estimators = jax.lax.scan(scan_fun, None, ks, length=ks.shape[0])

    estimators, _ = mpi.mpi_allgather_jax(estimators)  # (nk.utils.mpi.n_nodes, num_k_vec)
    estimators = jnp.mean(estimators, axis=0)  # (num_k_vec,)

    return estimators


@jax.jit
def _find_six_nearest_neighbours(rij: jnp.array, dist_rij: jnp.array):
    """ The Nj nearest neighbours (NN) of each i particle are found (for each
    of the M batches of MC samples). """
    Nj = 6  # number of NN
    rij_sorted = jax.vmap(sort_samples, in_axes=(0,0))(rij, dist_rij)  # (M,N,N,d)
    nn_rij = rij_sorted[:,:,1:Nj+1,:]  # don't take the case i=j (where trivially |rij|=0)
    return nn_rij


@jax.jit
def _director_field(rij: jnp.array, dist_rij: jnp.array):
    """ The director field is defined as \Psi(r_i)=\sum_{<j>}e^{-i*N_j*\theta_{ij})/N_j,
    where the sum is over the N_j nearest neighbours of particle i, and -Pi/2 <= \theta_{ij} <= Pi/2 is 
    the angle between r_{ij}=r_i-r_j and some chosen direction (taken here to be e_x=(1,0)).
    See for instance: https://en.wikipedia.org/wiki/Hexatic_phase. """
    Nj = 6  # number of NN
    d = rij.shape[-1]  # (M,N,N,d)
    nn_rij = _find_six_nearest_neighbours(rij, dist_rij)  # (M,N,Nj,d)
    # e_x = jnp.eye(d)[0]  # unit basis vector in the x-direction (d,)
    # theta_ij = _angle_btw_2_vectors(nn_rij, e_y)  # (M,N,Nj)
    theta_ij = jnp.arctan(nn_rij[...,1] / nn_rij[...,0])  # (M,N,Nj)  # project on e_x=(1,0) (the hexatic phase is only defined for d=2)
    Psi_ri = jnp.sum(jnp.exp(1j * Nj * theta_ij), axis=-1) / Nj # (M,N)
    
    # ## Weighting by the distance between particle i and its neighbours {j}
    # nn_dist_rij = jnp.sort(dist_rij, axis=-1)[:,:,1:Nj+1]  # (M,N,Nj)
    # Psi_ri = jnp.sum(nn_dist_rij * jnp.exp(1j * Nj * theta_ij), axis=-1) / jnp.sum(nn_dist_rij, axis=-1) # (M,N)
    
    return Psi_ri

from netket import jax as nkjax
def smooth_cutoff_to_boolean(A: jnp.array, cutoff: float, sharpness=3.0, key=None) -> jnp.array:
    """
    Apply a smooth cutoff to the array A using a sigmoid function and then
    convert the resulting smooth mask to a boolean mask using a probabilistic approach.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    # Compute the smooth mask using a sigmoid function
    smooth_mask = 1 / (1 + jnp.exp(-sharpness * (A - cutoff)))
    subkey = nkjax.mpi_split(key)
    random_values = jax.random.uniform(subkey, shape=A.shape)
    boolean_mask = smooth_mask > random_values
    return boolean_mask

@jax.jit
def _director_field_based_on_dist(rij: jnp.array, dist_rij: jnp.array, dist_cutoff: float):
    """ The director field is defined as \Psi(r_i)=\sum_{<j>}e^{-i*N_j*\theta_{ij})/N_j,
    where the sum is over the N(i) nearest neighbours of particle i, as defined by the 
    chose distance cutoff, and -Pi <= \theta_{ij} <= Pi is the angle between r_{ij}=r_i-r_j 
    and some chosen direction (taken here to be e_x=(1,0)). 
    See for instance: https://en.wikipedia.org/wiki/Hexatic_phase. """
    
    ## Theta/Heaviside function based mask
    mask = dist_rij < dist_cutoff # (M,N,N)
    
    # ## Smooth mask
    # mask = smooth_cutoff_to_boolean(dist_rij, dist_cutoff, sharpness=100.)
    
    num_near_neighs = jnp.sum(mask, axis=-1)  # (M,N)
    ## (M,N,N,d) -> (M,N,N) -> (M,N,N(i)) -> (M,N)
    Psi_ri = jnp.sum(jnp.where(mask, jnp.exp(1j * 6 * jnp.arctan2(rij[...,1], rij[...,0])), 0.), axis=-1) 
    Psi_ri /= num_near_neighs #6 # jnp.mean(num_near_neighs)  # (M,N)
    return Psi_ri


@jax.jit
def _hexatic_orientational_order_param_expect(
    mc_samples: jnp.array,
    L: jnp.array,
    dr: float,
    r: float,
) -> float:
    """ Compute g_6(r)=<\sum_{i<j} \Psi^*(r_i) \Psi(r_j) \delta(r-|r_{ij}|)>, where 
    \Psi(r_i) is the director field (defined above). """
    sigma = dr  # take the r_axis spacing for the Gaussian width
    _, N, d = mc_samples.shape  # (M,N,d)
    
    # epsilon = 0.1  # activation threshold associated with the kernel modeling the delta function

    # rij = mic_distance(mc_samples, L)  # (M,N,N,d)
    # dist_rij = jnp.linalg.norm(rij, axis=-1)  # (M,N,N)
    
    # delta_rij_minus_r = jnp.exp(-0.5 * (r - dist_rij)**2 / sigma**2)  # take a Gaussian kernel (M,N,N)
    # delta_rij_minus_r /= (sigma * jnp.sqrt(2 * jnp.pi))  # 1D Gaussian normalization
    # fill_utri_with_0 = jax.vmap(lambda x: jnp.tril(x, k=0), in_axes=(0,))
    # delta_rij_minus_r = fill_utri_with_0(delta_rij_minus_r)  # fill the upper triangular part with zeros to avoid overcounting pairs

    # Psi_ri = _director_field(rij, dist_rij)  # (M,N)
    # # rm = 2.9673  # [A]
    # # dist_cutoff = 4. / rm  # [dimensionless]
    # # Psi_ri = _director_field_based_on_dist(rij, dist_rij, dist_cutoff=dist_cutoff)  # (M,N)
    # Psi_rj_conj = jnp.conjugate(Psi_ri)  # (M,N)

    # estimators = jnp.einsum('...i,...j,...ij->...', Psi_ri, Psi_rj_conj, delta_rij_minus_r)  # (M,)

    # num_nz_contributions = jnp.sum(delta_rij_minus_r > epsilon, axis=(1,2))  # (M,)
    # estimators /= num_nz_contributions  # (M,)
    # # estimators /= (0.5 * N * (N-1))  # to take the average since we're performing \sum_{i<j}
    # # estimators /= N**2  # to take the average since we're performing \sum_{ij}
    # # estimators /= N  # since g(0)=\sum_i |\psi(ri)|^2, we need 1/N to have something of O(1)
    # estimators = jnp.mean(estimators)  # (M,)->1


    ## Test without using the Gaussian kernel:
    """ Compute g6(r) = <1/N_nz \sum_{i<j|r<=|r_i-r_j|<r+dr} \Psi^*(r_i) \Psi(r_j)>, where N_nz is the number
    of non-zero term in the sum, and the <...> denotes an average over many Monte Carlo configurations (denoted M).
    """

    rij = mic_distance(mc_samples, L)  # (M,N,N,d)  ## MANY THINGS CHOULD BE COMPUTED ONCE! OUTSIDE THE VMAP
    dist_rij = jnp.linalg.norm(rij, axis=-1)  # (M,N,N) 
    get_upper_triang = jax.vmap(lambda x: x[jnp.triu_indices(n=N,k=1)], in_axes=(0,))

    Psi_ri = _director_field(rij, dist_rij)  # (M,N)
    Psi_rj_conj = jnp.conjugate(Psi_ri)  # (M,N)
    g6 = jnp.einsum('ij,ik->ijk', Psi_ri, Psi_rj_conj)  # (M,N,N)
    g6 = get_upper_triang(g6).flatten()  # (M*N(N-1)/2)

    dist_rij = get_upper_triang(dist_rij).flatten()  # (M*N(N-1)/2,)
    mask = jnp.logical_and(r <= dist_rij, dist_rij < r + dr) # (M*N(N-1)/2,)
    num_nz_contributions = jnp.sum(mask)

    ## Average over the non-zero contributions within the N(N-1)/2 pairs in each of the M independent MC samples
    estimators = jnp.sum(jnp.where(mask, g6, 0.)) / num_nz_contributions

    return estimators


@expect.dispatch
def hexatic_orientational_order_param_expect(
    vstate: MCState,
    op: HexaticOrientationalOrderParameter,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    d = 2  # the hexatic phase is only defined for two-dimensional systems
    rs = op._radius_axis  # (num_radii,)
    dr = op._dr
    L = jnp.array(op.hilbert.extent)

    samples = vstate.samples  # (n_chains_per_rank,n_samples_per_chain,N*d)
    N = samples.shape[-1] // d  # number of particles
    samples = jnp.squeeze(samples).reshape(-1,N,d)  # (n_chains_per_rank*n_samples_per_chain,N,d)

    estimators = jax.vmap(_hexatic_orientational_order_param_expect, in_axes=(None, None, None, 0))(
        samples, L, dr, rs
    )  # (num_radii,)
    estimators, _ = mpi.mpi_allgather_jax(estimators)  # (nk.utils.mpi.n_nodes, num_radii)
    estimators = jnp.mean(estimators, axis=0)  # (num_radii,)

    return estimators


@jax.jit
def _hexatic_oop_fourier_expect(mc_samples: jnp.array, L: jnp.array, k: jnp.array) -> float:
    """ Compute S_6(k)=\nu_k * (\nu_k)^*/2, where \nu_k=\sum_{i=1}^N e^{i*k.r_i} \Psi(r_i),
    where \Psi(r_i) is the director field (defined above). """
    assert L.size == k.size  # L, k should have shape (d,)
    rij = mic_distance(mc_samples, L)  # (M,N,N,d)
    dist_rij = jnp.linalg.norm(rij, axis=-1)  # (M,N,N)
    Psi_ri = _director_field(rij, dist_rij)  # (M,N)
    exp_ikr = jnp.exp(1j * jnp.einsum('k,ijk->ij', k, mc_samples))  # (M,N)
    nu_k = jnp.sum(exp_ikr * Psi_ri, axis=-1)  # (M,)
    return jnp.mean(jnp.conjugate(nu_k) * nu_k)  # (M,) -> 1


@expect.dispatch
def hexatic_oop_fourier_expect(
    vstate: MCState,
    op: HexaticOOPFourier,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    ks = op._wavevectors  # (num_k_vec,d)
    d = ks.shape[-1]  # spatial dimension
    L = jnp.array(op.hilbert.extent)
    assert d == 2  # the hexatic phase is only defined for two-dimensional systems

    samples = vstate.samples  # (n_chains_per_rank,n_samples_per_chain,N*d)
    N = samples.shape[-1] // d  # number of particles
    samples = jnp.squeeze(samples).reshape(-1,N,d)  # (n_chains_per_rank*n_samples_per_chain,N,d)

    estimators = jax.vmap(_hexatic_oop_fourier_expect, in_axes=(None, None, 0))(samples, L, ks)  # (num_k_vec,)
    estimators, _ = mpi.mpi_allgather_jax(estimators)  # (nk.utils.mpi.n_nodes, num_k_vec)
    estimators = jnp.mean(estimators, axis=0)  # (num_k_vec,)

    return estimators / N


@jax.jit
def _pair_correlation_function_expect(mc_samples: jnp.array, L: jnp.array, r: jnp.array, sigma: float=0.1) -> float:
    """ Compute g^{(2)}(r) * V * n^2 = \sum_{i,j=1 | i!=j}^N \delta(r_{ij}-r), where r_{ij}=r_i-r_j, with {r_i} the MC samples,
    and r is a vector in the simulation cell (V is the volume and n the number density). We approximate the 
    delta function with a peaked Gaussian distribution. Assume `mc_samples` has shape (M,N,d) and `r` has shape (d,).
    Since here [r]=dimensionless, a factor of r_m^d (arising from sigma) should be introduced to get g^{(2)}(r). """
    _, N, d = mc_samples.shape
    rij = mic_distance(mc_samples, L)  # (M,N,N,d) 
    rij_minus_r = rij - r  # (M,N,N,d)
    rij_minus_r = jnp.remainder(rij_minus_r + L / 2.0, L) - L / 2.0
    rij_minus_r = jax.vmap(lambda x: jnp.concatenate((x[jnp.triu_indices(N,1)], x[jnp.tril_indices(N,-1)])), in_axes=(0,))(rij_minus_r)  # avoid the diagonal (M,N(N-1),d)
    estimators = jnp.exp(-0.5 * jnp.linalg.norm(rij_minus_r, axis=-1)**2 / sigma**2)  # (M,N(N-1)) [sigma]=[r]=dimensionless
    estimators /= (sigma**d * (2 * jnp.pi)**(d/2))  # [sigma]^d = dimensionless
    estimators = jnp.sum(estimators, axis=-1)  # (M,N(N-1)) -> (M,)
    return jnp.mean(estimators)  # (M,)->1


@expect.dispatch
def pair_correlation_function_expect(
    vstate: MCState,
    op: PairCorrelationFunction,
    chunk_size: Optional[int],
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    rs = op._grid_points  # (num_points[0]*...*num_points[d-1],d)
    d = rs.shape[-1]  # spatial dimension
    L = jnp.array(op.hilbert.extent)

    samples = vstate.samples  # (n_chains_per_rank,n_samples_per_chain,N*d)
    N = samples.shape[-1] // d  # number of particles
    samples = jnp.squeeze(samples).reshape(-1,N,d)  # (n_chains_per_rank*n_samples_per_chain,N,d)

    # estimators = jax.vmap(_pair_correlation_function_expect, in_axes=(None, None, 0))(samples, L, rs)  # (num_points[0]*...*num_points[d-1],)
    estimators = vmap_chunked(_pair_correlation_function_expect, in_axes=(None, None, 0), chunk_size=chunk_size)(samples, L, rs)
    estimators, _ = mpi.mpi_allgather_jax(estimators)  # (nk.utils.mpi.n_nodes, num_points[0]*...*num_points[d-1])
    estimators = jnp.mean(estimators, axis=0)  # (num_points[0]*...*num_points[d-1],)

    return estimators
