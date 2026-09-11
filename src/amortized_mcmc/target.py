"""True model densities used by the exact samplers.

This module is intentionally separate from the amortizer networks. Proposal
parameters may affect transition efficiency, but never enter these target
densities or the MH acceptance correction.
"""

from __future__ import annotations

import jax.numpy as jnp

from .graph import Graph
from .csp import CSPState


def poisson_log_likelihood(X: jnp.ndarray, L: jnp.ndarray, F: jnp.ndarray, alpha: jnp.ndarray, p_of_s: jnp.ndarray) -> jnp.ndarray:
    """Return the unnormalized Poisson log likelihood under the intensity link."""
    eta = L @ F.T + alpha[p_of_s, None]
    return jnp.sum(X * eta - jnp.exp(eta))


def laplace_mrf_logprob(L: jnp.ndarray, graph: Graph, scale: float = 1.0, edge_weight: float = 1.0) -> jnp.ndarray:
    """Evaluate the edge-preserving Laplace MRF log prior for ``L``."""
    differences = L[graph.senders] - L[graph.receivers]
    # Directed edges are intentional; callers can pass half-weight if using both directions.
    return -edge_weight * jnp.sum(jnp.sqrt(jnp.sum(differences * differences, axis=-1) + 1e-8)) / scale


def log_pi_L(
    L: jnp.ndarray,
    X: jnp.ndarray,
    F: jnp.ndarray,
    alpha: jnp.ndarray,
    p_of_s: jnp.ndarray,
    graph: Graph,
    mrf_scale: float = 1.0,
) -> jnp.ndarray:
    """Evaluate the unnormalized posterior block density for spatial factors."""
    return poisson_log_likelihood(X, L, F, alpha, p_of_s) + laplace_mrf_logprob(L, graph, mrf_scale)


def csp_log_prior_F(F: jnp.ndarray, csp: CSPState) -> jnp.ndarray:
    """Evaluate the Horseshoe-local/CSP-column prior for gene loadings."""
    precision = csp.phi[None, :] / (csp.lam * csp.lam)
    gaussian = 0.5 * jnp.sum(jnp.log(precision) - precision * F * F)
    local_half_cauchy = -jnp.sum(jnp.log1p(csp.lam * csp.lam))
    return gaussian + local_half_cauchy


def log_pi_F(F: jnp.ndarray, csp: CSPState) -> jnp.ndarray:
    """Return the loading prior density without an observation likelihood."""
    return csp_log_prior_F(F, csp)


def log_pi_F_joint(
    F: jnp.ndarray,
    X: jnp.ndarray,
    L: jnp.ndarray,
    alpha: jnp.ndarray,
    p_of_s: jnp.ndarray,
    csp: CSPState,
) -> jnp.ndarray:
    """Evaluate the true conditional target for ``F`` in a joint chain."""
    return poisson_log_likelihood(X, L, F, alpha, p_of_s) + csp_log_prior_F(F, csp)
