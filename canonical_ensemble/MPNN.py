import jax.numpy as jnp
import flax.linen as nn

import numpy as np
from typing import Callable, Union, Tuple, Any

from jax.nn.initializers import lecun_normal


class MLP(nn.Module):
    out_dim: int
    hidden_layers: Tuple[int]
    activation: Callable = nn.gelu
    output_activation: Callable = None
    last_bias: bool = True
    last_linear: bool = True
    dtype: Any = float
    kwargs: dict = None

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        kwargs = self.kwargs if self.kwargs is not None else {}
        dims = [in_dim, *self.hidden_layers, self.out_dim]
        for k in range(len(dims) - 1):
            last = k + 2 == len(dims)
            bias = not last or self.last_bias or not self.last_linear
            x = nn.Dense(dims[k + 1], use_bias=bias, param_dtype=self.dtype,
                           name=f'linear{k+1}', **kwargs)(x)
            if self.output_activation is not None:
                x = self.output_activation(x)
            if not last or not self.last_linear and not self.output_activation:
                x = self.activation(x)
        return x


class MLPFactory:
    def __init__(self, embedding_dim, intermediate_dim, mlp_layers):
        """ To instantiate MLPs.
        Args:
            intermediate_dim: dimension of the hidden layers
            embedding_dim: dimension of the output layer
            mlp_layers: number of hidden layers
        """
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.mlp_layers = mlp_layers

    def build_mlp(self):
        return MLP(self.embedding_dim, (self.intermediate_dim,) * self.mlp_layers)


class MPNNLayer(nn.Module):
    """ Implement one message passing iteration/layer. Here we assume a complete graph, as seen from the sum over all
    messages M_{ij} (for all j for a fixed i). The messages are calculated with an attention mechanism. """
    mlp_factory: Callable
    n_up: int
    n_down: int
    attention_dim: int

    @nn.compact
    def __call__(self, yi, Yij, Iij, i):
        """
        Args:
            yi (Array): vertex states
            Yij (Array): edge states
            Iij (Array): (static) input tensor containing physical information
            i (int): message passing iteration/layer

        Returns:
            yi (Array): updated vertex states
            Yij (Array): updated edge states
        """

        ## Instantiate the MLPs
        mlp1 = self.mlp_factory.build_mlp()
        mlp2 = self.mlp_factory.build_mlp()
        mlp3 = self.mlp_factory.build_mlp()
        mlp4 = self.mlp_factory.build_mlp()

        ## Compute the attention weights
        wquery = self.param(f'query_{i}', lecun_normal(), (Yij.shape[-1], self.attention_dim), np.float64)
        wkey = self.param(f'key_{i}', lecun_normal(), (Yij.shape[-1], self.attention_dim), np.float64)
        xquery = jnp.dot(Yij, wquery)
        xkey = jnp.dot(Yij, wkey)
        attention_weights = mlp1(jnp.einsum('...ijk,...jlk->...ilk', xquery, xkey) / jnp.sqrt(self.attention_dim))

        ## Compute the messages
        Mij = attention_weights * mlp2(Yij)

        ## Update the vertex states by aggregating (i.e. summing here) the messages coming at each vertex
        yi = mlp3(jnp.concatenate((yi, jnp.sum(Mij, axis=-2)), axis=-1))

        ## Update the edge states
        Yij = jnp.concatenate((Iij, mlp4(jnp.concatenate((Yij, Mij), axis=-1))), axis=-1)

        return yi, Yij


class MPNN(nn.Module):
    n_up: int
    n_down: int
    L: Union[jnp.array, float]
    embedding_dim: int
    intermediate_dim: int
    mlp_layers: int
    attention_dim: int
    n_features: int
    n_interactions: int

    @nn.compact
    def __call__(self, x):  
        """
        x: MC samples of shape (n_samples, n_particles, spatial_dimension)
        """
        assert len(x.shape) == 3
        M, N, sdim = x.shape

        ## Instantiate the network factory to generate the MLPs
        mlp_factory = MLPFactory(self.embedding_dim, self.intermediate_dim, self.mlp_layers)

        ## Initialize 2 hidden variables to be optimized 
        embedding_func = nn.Embed(
            num_embeddings=2,  
            features=self.n_features,
            dtype=np.float64,
            param_dtype=np.float64,
            embedding_init=lecun_normal(),
        )
        hidden_vars = jnp.tile(embedding_func(jnp.array([0, 1]))[None, ...], (M, 1, 1))  # (M,num_embeddings,n_features)

        ## Compute coordinate differences, periodized coordinate differences and periodized distances
        rij = x[..., :, None, :] - x[..., None, :, :]  # (M,N,N,d)
        sin_cos_rij = jnp.concatenate((jnp.sin(2 * jnp.pi / self.L * rij), jnp.cos(2 * jnp.pi / self.L * rij)), axis=-1)
        # numerically more stable to add 1 on the vanishing diagonal entries before taking the norm
        d_sin_rij = jnp.linalg.norm(jnp.sin(jnp.pi / self.L * rij) + jnp.eye(N)[..., None], axis=-1, keepdims=True) 
        d_sin_rij *= (1. - jnp.eye(N)[..., None])  # revert back to 0's on the diagonal

        ## Initialize spin labels and compute s_i * s_j
        s = jnp.hstack((self.n_up * [1], self.n_down * [-1]))
        ss = jnp.outer(s, s)[..., None]

        ## Initialize vertex states y_i with one dynamic (h0) hidden variable 
        h0 = jnp.tile(hidden_vars[:, 0, :][:, None, :], (1, N, 1))  # (M,N,n_features)
        yi = h0  # (M,N,n_features)

        ## Initialize edge states Y_{ij} with the physical information (Iij) and one dynamic (H0) hidden variable
        Iij = jnp.concatenate((sin_cos_rij, d_sin_rij, jnp.tile(ss[None, :, :, :], (M, 1, 1, 1))), axis=-1)
        H0 = jnp.tile(hidden_vars[:, 1, :][:, None, None, :], (1, N, N, 1))  # (M,N,N,n_features)
        Yij = jnp.concatenate((Iij, H0), axis=-1)

        ## Perform message passing iterations
        for i in range(self.n_interactions):
            layer = MPNNLayer(mlp_factory, self.n_up, self.n_down, self.attention_dim)
            yi, Yij = layer(yi, Yij, Iij, i)
        return yi, Yij
