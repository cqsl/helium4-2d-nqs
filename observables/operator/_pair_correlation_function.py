from netket.operator._abstract_observable import AbstractObservable
import jax.numpy as jnp


class PairCorrelationFunction(AbstractObservable):
    def __init__(
        self,
        hilbert: None,
        grid_points: jnp.array,
    ):
        super().__init__(hilbert)
        self._grid_points = grid_points
        self._num_points = grid_points.shape[0]

    def __repr__(self):
        return f"PairCorrelationFunction(hilbert={self.hilbert}, L={self.hilbert.extent}, num_points={self._num_points})"
