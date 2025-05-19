import jax.numpy as jnp
from typing import Tuple
import jax


@jax.jit
def coordinate_differences(r):
    """ Compute the coordinate differences rij = ri - rj, with `r` of shape (N,d) or (M,N,d)."""
    return r[...,None, :, :] - r[...,:, None, :]

@jax.jit
def mic_distance(x: jnp.array, L: jnp.array):
    """Compute distances between particles using the minimum image convention.
    Take `x` of shape (M,N,d) and `L` of shape (d,)."""
    distances = x[:, jnp.newaxis, :, :] - x[:, :, jnp.newaxis, :]
    distances = jnp.remainder(distances + L / 2.0, L) - L / 2.0
    return distances  # (M,N,N,d)


@jax.jit
def sort_samples(R: jnp.array, dist: jnp.array) -> jnp.array:
    """ `R` has shape (X,N,d) and `dist` has shape (X,N). This function
    sorts the 2nd dimension of `R`, corresponding to the particle index, 
    by increasing `dist` values. In the code below X:=M. """
    return R[jnp.arange(R.shape[0])[:,None], jnp.argsort(dist, axis=-1)]  # (X,N,d)


@jax.jit
def _angle_btw_2_vectors(v1: jnp.array, v2: jnp.array):
    """ `v1` and `v2` should share the same dimension in the last axis. """
    v1_dot_v2 = jnp.einsum('...i,...i->...', v1, v2)
    v1_norm = jnp.linalg.norm(v1, axis=-1)
    v2_norm = jnp.linalg.norm(v2, axis=-1)
    jnp.arccos(v1_dot_v2/(v1_norm * v2_norm))  # 0 <= theta <= Pi (...)
    return 


def rectangular_grid(box_size: jnp.array, num_coords: jnp.array) -> jnp.array:
    d = box_size.shape[0]
    assert num_coords.size == d
    coord_axes = [jnp.linspace(box_size[i][0], box_size[i][1], num_coords[i]) for i in range(d)]  # list of d sub-arrays of shape (num_coords[i],)
    # coord_axes = [jnp.linspace(0, box_size[i], num_coords[i]) for i in range(d)]  # list of d sub-arrays of shape (num_coords[i],)
    coord_grid = jnp.meshgrid(*coord_axes, indexing='ij')  # tuple with d sub-arrays of shape num_coords=(num_coords[0],...,num_coords[d-1])
    grid_points = jnp.vstack([grid.flatten() for grid in coord_grid]).T  # (num_coords[0]*...*num_coords[d-1],d)
    grid_spacings = []
    for axis in coord_axes: 
        grid_spacings.append(axis[1]-axis[0])
    grid_spacings = jnp.array(grid_spacings)
    return grid_spacings, grid_points


def spherical_grid(box_size: jnp.array, max_radius: float, num_points: Tuple[int,...]) -> jnp.array:
    if len(box_size) == 2:
        num_r, num_phi = num_points
        r_axis = jnp.linspace(0, max_radius, num=num_r)
        phi_axis = jnp.linspace(0, 2*jnp.pi, num=num_phi, endpoint=False)  # don't take phi=2*pi
        angular_coords = jnp.vstack((jnp.cos(phi_axis), jnp.sin(phi_axis))).T  # (num_phi, 2)
        angular_coords = jnp.tile(angular_coords[None], reps=(num_r,1,1))  # (num_r, num_phi, 2)
        grid_points = jnp.einsum('i,ijk->ijk', r_axis, angular_coords)  # (num_r, num_phi, 2)
        grid_points = grid_points.reshape(-1, grid_points.shape[-1])  # (num_r * num_phi, 2)
        return grid_points
    elif len(box_size) == 3:
        num_r, num_phi, num_theta = num_points
        r_axis = jnp.linspace(0, max_radius, num=num_r)
        phi_axis = jnp.linspace(0, 2*jnp.pi, num=num_phi, endpoint=False)  # don't take phi=2*pi
        theta_axis = jnp.linspace(0, jnp.pi, num=num_theta)
        x = jnp.outer(jnp.sin(theta_axis), jnp.cos(phi_axis)).flatten()
        y = jnp.outer(jnp.sin(theta_axis), jnp.sin(phi_axis)).flatten()
        z = jnp.outer(jnp.cos(theta_axis), jnp.ones(num_phi)).flatten()
        angular_coords = jnp.vstack((x, y, z)).T  # (num_phi * num_theta, 3)
        angular_coords = jnp.tile(angular_coords[None], reps=(num_r,1,1))  # (num_r, num_phi * num_theta, 3)
        grid_points = jnp.einsum('i,ijk->ijk', r_axis, angular_coords)  # (num_r, num_phi * num_theta, 3)
        grid_points = grid_points.reshape(-1, grid_points.shape[-1]) # (num_r * num_phi * num_theta, 3)
        return grid_points
    else:
        raise ValueError(f"A grid in {len(box_size)} dimensional space is not supported.")


def cubic_wavevector_grid(box_size: jnp.array, d: int, n_max=10) -> jnp.array:
    """ Cubic grid in reciprocal space. """
    n_step = 1  # because of PBCs, the wavevectors are discrete
    n_1d_axis = jnp.arange(-n_max, n_max + n_step, n_step)
    n_axes = jnp.tile(n_1d_axis[None], reps=(d,1))
    ns_grid = jnp.meshgrid(*n_axes)
    ns = jnp.vstack(ns_grid).reshape(d,-1).T  # ((2*n_max+1)**d, d)
    ks = 2 * jnp.pi * ns / box_size  # 1/[L]
    return ks

