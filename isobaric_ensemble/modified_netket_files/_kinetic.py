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
from typing import Optional, Callable, Union, Tuple
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils.types import DType, PyTree, Array
import netket.jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import HashableArray


def jacrev(f):
    def jacfun(x):
        y, vjp_fun = nkjax.vjp(f, x)
        if y.size == 1:
            eye = jnp.eye(y.size, dtype=x.dtype)[0]
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        else:
            eye = jnp.eye(y.size, dtype=x.dtype)
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        return J

    return jacfun


def jacfwd(f):
    def jacfun(x):
        jvp_fun = lambda s: jax.jvp(f, (x,), (s,))[1]
        eye = jnp.eye(len(x), dtype=x.dtype)
        J = jax.vmap(jvp_fun, in_axes=0)(eye)
        return J

    return jacfun


class KineticEnergy(ContinuousOperator):
    r"""This is the kinetic energy operator (hbar = 1). The local value is given by:
    :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        arguments: tuple[float] = (1.,1.,1.),
        dtype: Optional[DType] = None,
    ):
        r"""Args:
        hilbert: The underlying Hilbert space on which the operator is defined
        arguments: data to be passed when computing the kinetic energy
        dtype: Data type of the matrix elements. Defaults to `np.float64`
        """

        self.arguments = arguments
        self._is_hermitian = True
        self.__attrs = None

        super().__init__(hilbert, float)

    @property
    def mass(self):
        return self._mass

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel_single(
        self, logpsi: Callable, params: PyTree, x: Array, arguments: Optional[PyTree]
    ):
        ## `x` are samples in the box [0,1]x[0,1]
        Lx, Ly, theta = arguments  # [A], [A], rad
        d = 2  # spatial dimensions
        N = x.shape[-1] // d  # number of particles

        def logpsi_x(x):
            return logpsi(params, x)

        dlogpsi_x = jacrev(logpsi_x)

        def kin_func(x, Lx, Ly, theta):
            inv_jacobian = jnp.array([[1/Lx, -jnp.cos(theta)/(jnp.sin(theta) * Lx)], [0., 1/(jnp.sin(theta) * Ly)]])  # (d,d) [A] 
            
            grad = dlogpsi_x(x)[0][0].reshape(N,d)  # (N,d)
            grad = jnp.einsum('ij,...j->...i', jnp.transpose(inv_jacobian), grad).flatten()  # (N*d,)
            dp_dx = jnp.sum(grad**2)  # (N*d,) -> 1

            hess_transf_mat = jnp.matmul(inv_jacobian, jnp.transpose(inv_jacobian))  # (d,d)
            hessian = jacfwd(dlogpsi_x)(x)[0].reshape(N*d,N*d)  # (N*d,N*d)

            dp_dx2 = 0.
            for i in range(0,N*d,d):
                dp_dx2 += jnp.sum(hess_transf_mat * hessian[i:i+d,i:i+d])  # (d,d)*(d,d) -> 1
            
            return dp_dx2 + dp_dx

         ## Kinetic energy
        kin = kin_func(x, Lx, Ly, theta)

        ## Derivative of the kinetic energy w.r.t the Lx and Ly parameters
        dkin_dLx, dkin_dLy, dkin_dtheta = jax.grad(kin_func, argnums=(1,2,3))(x, Lx, Ly, theta)  # scalar [A^{-1}], [A^{-1}], [rad^{-1}]

        # dkin_dtheta = 0.

        return kin, (dkin_dLx, dkin_dLy, dkin_dtheta) 

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, arguments: Optional[PyTree]
    ):
        return self._expect_kernel_single(logpsi, params, x, arguments)

    def _pack_arguments(self) -> PyTree:
        return self.arguments

    @property
    def _attrs(self):
        if self.__attrs is None:
            self.__attrs = (self.hilbert, HashableArray(self.arguments))
        return self.__attrs

    def __repr__(self):
        return f"KineticEnergy(arguments={self.arguments})"