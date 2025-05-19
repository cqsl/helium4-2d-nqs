# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple
from collections.abc import Hashable

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import struct, HashableArray

import jax
import jax.numpy as jnp
from functools import partial


@struct.dataclass
class PotentialOperatorPyTree:
    """Internal class used to pass data from the operator to the jax kernel.

    This is used such that we can pass a PyTree containing some static data.
    We could avoid this if the operator itself was a pytree, but as this is not
    the case we need to pass as a separte object all fields that are used in
    the kernel.

    We could forego this, but then the kernel could not be marked as
    @staticmethod and we would recompile every time we construct a new operator,
    even if it is identical
    """

    potential_fun: Callable = struct.field(pytree_node=False)
    coefficient: Array


class PotentialEnergy(ContinuousOperator):
    r"""Returns the local potential energy defined in afun"""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        afun: Callable,
        pressure: float = 1.,
        arguments: tuple[float] = ( 1.,1.),
        dtype: Optional[DType] = float,
    ):
        r"""
        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            afun: The potential energy as function
            pressure: The pressure of the system in units of KA^{-2}
            arguments: data to be passed when computing the potential energy
            dtype: Data type of the matrix elements. Defaults to `np.float64`
        """

        self._afun = afun
        self.arguments = arguments
        self.__attrs = None
        self._pressure = pressure

        super().__init__(hilbert, float)

    @property
    def coefficient(self) -> Array:
        return self._coefficient

    @property
    def is_hermitian(self) -> bool:
        return True

    def _expect_kernel_single(
        self, logpsi: Callable, params: PyTree, x: Array, arguments: Optional[PyTree]
    ):
        Lx, Ly, theta = arguments  # [A], [A], [rad]
        pressure = self._pressure  # [KA^{-2}] 

        ## Potential energy
        pot = self._afun(x, pressure, Lx, Ly, theta)  # scalar [dimensionless]

        ## Derivative of the potential energy w.r.t the Lx and Ly parameters
        dpot_dLx, dpot_dLy, dpot_dtheta = jax.grad(self._afun, argnums=(2,3,4))(x, pressure, Lx, Ly, theta)  # scalar [A^{-1}], [A^{-1}], [rad^{-1}]

        # dpot_dtheta = 0.

        return pot, (dpot_dLx, dpot_dLy, dpot_dtheta)

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, arguments: Optional[PyTree]
    ):
        return self._expect_kernel_single(logpsi, params, x, arguments)

    def _pack_arguments(self):
        return self.arguments

    @property
    def _attrs(self) -> tuple[Hashable, ...]:
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self._afun,
                HashableArray(self.arguments),
            )
        return self.__attrs

    def __repr__(self):
        return f"Potential(arguments={self.arguments}, function={self._afun})"