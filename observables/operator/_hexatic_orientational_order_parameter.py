from netket.operator._abstract_observable import AbstractObservable
import jax.numpy as jnp


class HexaticOrientationalOrderParameter(AbstractObservable):
    def __init__(
        self,
        hilbert: None,
        radius_axis: jnp.array,
        dr: float,
    ):
        super().__init__(hilbert)
        self._radius_axis = radius_axis
        self._dr = dr

    def __repr__(self):
        return f"HexaticOrientationalOrderParameter(hilbert={self.hilbert})"
