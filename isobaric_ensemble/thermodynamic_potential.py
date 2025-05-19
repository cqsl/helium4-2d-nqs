import jax.numpy as jnp
import scipy.integrate as integrate
from jax.scipy.integrate import trapezoid
from jax.scipy.special import gamma


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
    eps = 1.
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

def potential_nearest_image(x: jnp.array, L: jnp.array, theta: float) -> float:
    """ The `x` fed to the potential, of shape (N*d,) are dimensionless and in the
    box [0,1]x[0,1]. We need to transform the coordinates in the physical box [0,Lx]x[0,Ly]
    to make them dimnesionful (in [A]), calculate the interparticle distances (in [A]) and
    then divide the distances by the Aziz length scale (rm=2.9673 A) to make them dimensionless.
    Then we can feed the dimensionless distances to the Aziz potential. """
    rm = 2.9673  # [A]
    d = L.size  # spatial dimension
    x = x.reshape(-1, d)  # (N,d)
    N = x.shape[0]
    # x *= L  # [A]  # in the specific case where theta=pi/2
    Lx, Ly = L  # [A]
    jacobian = jnp.array([[Lx, jnp.cos(theta) * Ly],[0., jnp.sin(theta) * Ly]])  # (d,d) [A]
    x = jnp.einsum('ij,...j->...i', jacobian, x)  # apply the jacobian to each single-particle coordinate (N,d) [A]

    ## Potential over the nearest image (using either the minimum image convention distance or the sine distance)
    Lmin = jnp.min(L)  # [A]
    rc = Lmin/2  # [A]  
    v_rc = aziz(rc/rm) 
    distances = min_imag_conv_distance(x, L)  # (N(N-1)/2,) [A]
    v = jnp.sum(jnp.where(distances <= rc, aziz(distances/rm) - v_rc, 0.))  
    
    ## Constant potential contributions (importantly they depend on the box size)
    density = N / jnp.prod(L)  # [A^-d]
    v += v_shift_fun(N, density * rm**d, d, rc/rm)
    v += v_tail_fun(N, density * rm**d, d, rc/rm)

    return v  # [dimensionless]


def potential_fully_periodized(x: jnp.array, L: jnp.array, theta: float) -> float:
    """ The `x` fed to the potential, of shape (N*d,) are dimensionless and in the
    box [0,1]x[0,1]. We need to transform the coordinates in the physical box [0,Lx]x[0,Ly]
    to make them dimnesionful (in [A]), calculate the interparticle distances (in [A]) and
    then divide the distances by the Aziz length scale (rm=2.9673 A) to make them dimensionless.
    Then we can feed the dimensionless distances to the Aziz potential. """
    rm = 2.9673  # [A]
    d = L.size  # spatial dimension
    x = x.reshape(-1, d)  # (N,d)
    N = x.shape[0]
    # x *= L  # [A]  # in the specific case where theta=pi/2
    Lx, Ly = L  # [A]
    jacobian = jnp.array([[Lx, jnp.cos(theta) * Ly],[0., jnp.sin(theta) * Ly]])  # (d,d) [A]
    x = jnp.einsum('ij,...j->...i', jacobian, x)  # apply the jacobian to each single-particle coordinate (N,d) [A]

    ## Full periodized potential (over nearest neighbours (NN), next NN, up to some cutoff)
    rij = coordinate_diff(x)  # (N(N-1)/2,d) [A]
    ns = create_grid(d=d, n_max=2)  # (X,d):=((2*n_max+1)**d,d)
    # n_times_L = ns * L  # (X,d) [A]
    J_times_n = jnp.einsum('ij,...j->...i', jacobian, ns)  # (X,d) [A]
    rij_minus_J_times_n = rij[:, None, :] - J_times_n[None, :, :]  # (N(N-1)/2, X, d) 
    distances = jnp.linalg.norm(rij_minus_J_times_n, axis=-1).flatten()  # (N(N-1)/2 * X,)
    v = jnp.sum(aziz(distances/rm)) # periodic images
    
    ## Constant potential contributions (importantly they depend on the box size);
    ## If sufficiently many images are considered, the shift and tail contributions should be negligible.
    madelung_distances = jnp.linalg.norm(J_times_n[1:], axis=-1)  # don't sum over n=(0,0), i.e. `ns[0]`
    v_madelung = 0.5 * jnp.sum(aziz(madelung_distances/rm))
    v += N * v_madelung

    return v  # [dimensionless]

def gibbs_potential(x: jnp.array, pressure: float, L: jnp.array, theta: float) -> float:
    """ Effective Gibbs potential: V(R) + pV. """
    epsilon = 10.8  # [K]
    p_over_epsilon = pressure / epsilon  # [KA^{-d}]/[K]=[A^{-d}]
    v = potential_fully_periodized(x, L, theta)  # either "potential_nearest_image" or "potential_fully_periodized"
    v += p_over_epsilon * jnp.prod(L) * jnp.sin(theta)  # [A^{-d}*A^d]=[dimensionless]
    return v