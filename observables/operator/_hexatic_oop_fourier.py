from netket.operator._abstract_observable import AbstractObservable
import jax.numpy as jnp


class HexaticOOPFourier(AbstractObservable):
    def __init__(
        self,
        hilbert: None,
        wavevectors: jnp.array,
    ):
        super().__init__(hilbert)
        self._wavevectors = wavevectors
        self._num_k_vectors = wavevectors.shape[0]

    def __repr__(self):
        return f"HexaticOOPFourier(hilbert={self.hilbert}, L={self.hilbert.extent}, num_k_vectors={self._num_k_vectors})"
