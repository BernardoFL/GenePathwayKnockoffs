"""Gibbs update for the pathway-coupling strength ``rho``.

``rho`` is never touched by either amortized head; it moves only through
this conjugate Gamma full conditional, exactly like the other Gibbs blocks
in :mod:`amortized_mcmc.csp` and :mod:`amortized_mcmc.alpha`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .csp import CSPState
from .target import pathway_laplacian_rank, pathway_quadratic_forms


def gibbs_step_rho(
    key: jax.Array,
    F: jnp.ndarray,
    csp: CSPState,
    pathway_laplacian: jnp.ndarray,
    a_rho: float = 1.0,
    b_rho: float = 1.0,
) -> jnp.ndarray:
    """Draw ``rho`` from its Gamma conjugate conditional.

    The shape and rate use the Laplacian's pseudo-rank ``r`` (not ``G``,
    which would silently mis-normalize the update because the pathway
    Laplacian is rank deficient by one null direction per connected
    component) and the quadratic forms of only the *active* (slab) CSP
    columns, matching :func:`amortized_mcmc.target.pathway_logprob`.
    """
    active = csp.z == jnp.arange(csp.H)
    rank = pathway_laplacian_rank(pathway_laplacian)
    quadratic = pathway_quadratic_forms(F, pathway_laplacian)
    shape = a_rho + 0.5 * rank * jnp.sum(active)
    rate = b_rho + 0.5 * jnp.sum(jnp.where(active, quadratic, 0.0))
    return jax.random.gamma(key, shape) / rate
