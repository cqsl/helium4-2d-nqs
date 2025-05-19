from netket.operator._abstract_observable import AbstractObservable
import jax.numpy as jnp


class MomentumDistribution(AbstractObservable):
    def __init__(
        self,
        hilbert: None,
        grid_spacings: jnp.array,
        grid_points: jnp.array,
        wavevectors: jnp.array,
    ):
        super().__init__(hilbert)
        self._grid_spacings = grid_spacings
        self._grid_points = grid_points
        self._wavevectors = wavevectors
        self._num_r_points = grid_points.shape[0]
        self._num_k_vectors = wavevectors.shape[0]

    def __repr__(self):
        return f"MomentumDistribution(hilbert={self.hilbert}, L={self.hilbert.extent}, num_r_points={self._num_r_points}, num_k_vectors={self._num_k_vectors})"
