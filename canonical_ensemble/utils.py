import flax
import jax.numpy as jnp


def coord_diff(r):
    """ Compute the coordinate differences rij = ri - rj. """
    assert len(r.shape) == 2
    N, _ = r.shape
    dist = r[None, :, :] - r[:, None, :]
    return dist[jnp.triu_indices(N, 1)]

def find_k_largest_values(pytree, k=1):
    flat_tree = flax.traverse_util.flatten_dict(pytree)
    max_values = {k: jnp.max(v) for k, v in flat_tree.items()}
    largest_k_values = dict(sorted(max_values.items(), key=lambda x: x[1])[-k:])
    return largest_k_values 