import netket as nk
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from jax.nn.initializers import zeros
from typing import Optional, Tuple
from MPNN import MPNN
from utils import *
import jax


class McMillanWithBackflow(nn.Module):

    d: int
    n_up: int
    n_down: int
    L: Tuple[float,...]
    embedding_dim: int = 8
    intermediate_dim: int = 8
    mlp_layers: int = 1
    attention_dim: int = 8
    n_features: int = 8
    n_interactions: int = 1
    cusp_exponent: Optional[int] = 5
    
    @nn.compact
    def __call__(self, r): 

        flag = 1 if r.ndim == 1 else 0
        r = r[None] if r.ndim == 1 else r
        batch_shape = r.shape[0]  # assumes batched samples
        r = r.reshape(batch_shape, -1, self.d)
        _, N, _ = r.shape
        L = jnp.array(self.L)
        
        _, Y = MPNN(
            n_up=self.n_up,
            n_down=self.n_down,
            L=L,
            embedding_dim=self.embedding_dim, 
            intermediate_dim=self.intermediate_dim,
            mlp_layers=self.mlp_layers,
            attention_dim=self.attention_dim,
            n_features=self.n_features,
            n_interactions=self.n_interactions
        )(r)

        rij_up_tr = jax.vmap(coord_diff, in_axes=(0,))(r) # (M,N,d) -> (M,N(N-1)/2,d)
        dsin_rij = jnp.linalg.norm(0.5 * L * jnp.sin(jnp.pi * rij_up_tr / L), axis=-1) # (M,N(N-1)/2)

        Y = 0.5 * (Y + jnp.transpose(Y, axes=(0,2,1,3)))  # symmetrize Y w.r.t. the particle indices
        Yij = jax.vmap(lambda x: x[jnp.triu_indices(N, 1)], in_axes=(0,))(Y) # (M,N,N,d) -> (M,N(N-1)/2,d)

        delta1 = self.param("delta1", nn.initializers.constant(3.), (1,), float)
        cusp = -delta1 * jnp.sum(1/dsin_rij**self.cusp_exponent, axis=-1)

        delta2 = self.param("delta2", zeros, (1,), float)
        propagator = nk.models.MLP(hidden_dims=(self.embedding_dim,))(Yij)  # (M,N(N-1)/2,d) -> (M,N(N-1)/2)
        propagator = delta2 * jnp.sum(propagator, axis=-1)

        log_psi = cusp + propagator
        log_psi = log_psi[0] if flag == 1 else log_psi
        return log_psi