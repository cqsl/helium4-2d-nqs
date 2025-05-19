import netket as nk
import jax.numpy as jnp
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

    ## Potential over the nearest image (using either the minimum image convention distance or the sine distance)
    Lmin = jnp.min(L) 
    rc = Lmin/2  
    v_rc = aziz(rc) 
    distances = min_imag_conv_distance(x, L)  # (N(N-1)/2,)
    v = jnp.sum(jnp.where(distances <= rc, aziz(distances) - v_rc, 0.))  
    
    ## Constant potential contributions (importantly they depend on the box size)
    density = N / jnp.prod(L) 
    v += v_shift_fun(N, density, d, rc)
    v += v_tail_fun(N, density, d, rc)

    # ## Full periodized potential (over nearest neighbours (NN), next NN, up to some cutoff)
    # rij = coordinate_diff(x)  # (N(N-1)/2,d) 
    # ns = create_grid(d=d, n_max=1)  # (X,d)=((2*n_max+1)**d,d)
    # n_times_L = ns * L  # (X,d) 
    # rij_minus_n_times_L = rij[:, None, :] - n_times_L[None, :, :]  # (N(N-1)/2, X, d) 
    # distances = jnp.linalg.norm(rij_minus_n_times_L, axis=-1).flatten()  # (N(N-1)/2 * X,)
    # v = jnp.sum(aziz(distances)) # periodic images
    
    # ## Constant potential contributions (importantly they depend on the box size);
    # ## If sufficiently many images are considered, the shift and tail contributions should be negligible.
    # madelung_distances = jnp.linalg.norm(n_times_L[1:], axis=-1)  # don't sum over n=(0,0), i.e. `ns[0]`
    # v_madelung = 0.5 * jnp.sum(aziz(madelung_distances))
    # v += N * v_madelung

    return v  # [dimensionless]