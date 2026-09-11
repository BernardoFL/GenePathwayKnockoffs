"""NumPyro wrappers for evaluating the spatial target density."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import log_density

from .graph import Graph


def spatial_model(
    X: jnp.ndarray,
    F: jnp.ndarray,
    alpha: jnp.ndarray,
    p_of_s: jnp.ndarray,
    graph: Graph,
    mrf_scale: float = 1.0,
) -> None:
    """Define the L likelihood and Laplace-MRF target for NumPyro.

    The auxiliary standard-normal site is cancelled exactly by a factor so the
    resulting log density is the Poisson likelihood plus the Laplace MRF prior.
    It gives NumPyro's ``log_density`` a named parameter site for L without
    changing the intended target.
    The auxiliary standard-normal site is cancelled so this model's joint
    density is exactly the Poisson likelihood plus the MRF factor.
    """
    latent_shape = (X.shape[0], F.shape[1])
    base = dist.Normal(0.0, 1.0).expand(latent_shape).to_event(2)
    L = numpyro.sample("L", base)
    numpyro.factor("cancel_auxiliary_prior", -jnp.sum(base.log_prob(L)))
    eta = L @ F.T + alpha[p_of_s, None]
    numpyro.sample("X", dist.Poisson(jnp.exp(eta)).to_event(2), obs=X)
    differences = L[graph.senders] - L[graph.receivers]
    mrf = -jnp.sum(jnp.sqrt(jnp.sum(differences * differences, axis=-1) + 1e-8)) / mrf_scale
    numpyro.factor("laplace_mrf", mrf)


def numpyro_log_pi_L(
    L: jnp.ndarray,
    X: jnp.ndarray,
    F: jnp.ndarray,
    alpha: jnp.ndarray,
    p_of_s: jnp.ndarray,
    graph: Graph,
    mrf_scale: float = 1.0,
) -> jnp.ndarray:
    """Evaluate the unnormalized L target through NumPyro trace machinery."""
    log_joint, _ = log_density(
        spatial_model,
        (X, F, alpha, p_of_s, graph, mrf_scale),
        {},
        {"L": L},
    )
    return log_joint