from netket.operator._abstract_observable import AbstractObservable
import jax.numpy as jnp


class OneBodyDensityMatrix(AbstractObservable):
    def __init__(
        self,
        hilbert: None,
        displacement_vectors: jnp.array,
    ):
        super().__init__(hilbert)
        self._displacement_vectors = displacement_vectors
        self._num_points = displacement_vectors.shape[0]

    def __repr__(self):
        return f"OneBodyDensityMatrix(hilbert={self.hilbert}, L={self.hilbert.extent}, num_points={self._num_points})"
