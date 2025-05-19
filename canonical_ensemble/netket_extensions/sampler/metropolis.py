import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import netket as nk
from netket.utils import struct
from netket.sampler import MetropolisSampler
from netket.hilbert import ContinuousHilbert
from .rules import AdaptiveGaussianRule


class MetropolisGaussAdaptive(MetropolisSampler):
    """Metropolis sampler that adaptively changes the sampler scale to get a given target acceptance."""

    target_acceptance: float = 0.6
    sigma_limits: Any = None

    def __init__(self, *args, initial_sigma=1.0, target_acceptance=0.5, sigma_limits=None, **kwargs):
        rule = AdaptiveGaussianRule(initial_sigma=initial_sigma)
        if sigma_limits is None:
            sigma_limits = [initial_sigma * 1e-2, initial_sigma * 1e2]
        assert len(args) == 1, "should only pass hilbert"
        hilbert = args[0]
        args = [hilbert, rule]
        if not isinstance(hilbert, ContinuousHilbert):
            raise ValueError(
                f"This sampler only works for ContinuousHilbert Hilbert spaces, got {type(hilbert)}."
            )
        self.target_acceptance = target_acceptance
        self.sigma_limits = sigma_limits
        super().__init__(*args, **kwargs)

    def _sample_next(self, machine, parameters, state):
        new_state, new_σ = super()._sample_next(machine, parameters, state)

        if self.target_acceptance is not None:
            acceptance = new_state.n_accepted / new_state.n_steps
            sigma = new_state.rule_state
            new_sigma = sigma / (
                self.target_acceptance
                / jnp.max(jnp.stack([acceptance, jnp.array(0.05)]))
            )
            new_sigma = jnp.max(jnp.array([new_sigma, self.sigma_limits[0]]))
            new_sigma = jnp.min(jnp.array([new_sigma, self.sigma_limits[1]]))
            new_rule_state = new_sigma
            new_state = new_state.replace(rule_state=new_rule_state)

        return new_state, new_σ