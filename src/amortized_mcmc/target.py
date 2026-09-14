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


def standard_normal_logprob(value: jnp.ndarray) -> jnp.ndarray:
    """Return the summed standard-normal log density."""
    return -0.5 * jnp.sum(value * value + jnp.log(2.0 * jnp.pi))


def pathway_laplacian_from_affinity(affinity: jnp.ndarray) -> jnp.ndarray:
    """Build the graph Laplacian ``D - A`` from a symmetric affinity matrix."""
    degree = jnp.sum(affinity, axis=-1)
    return jnp.diag(degree) - affinity


def pathway_laplacian_rank(pathway_laplacian: jnp.ndarray, tol: float = 1e-8) -> jnp.ndarray:
    """Return the pseudo-rank of a (possibly rank-deficient) graph Laplacian.

    A connected-component graph Laplacian has one null eigenvalue per
    component, so its ordinary determinant is zero; the pathway prior's
    normalizing constant and ``rho``'s conjugate update must use this rank
    (the count of non-null eigen-directions), never ``G`` itself, or the
    normalization is silently wrong.
    """
    eigenvalues = jnp.linalg.eigvalsh(pathway_laplacian)
    return jnp.sum(eigenvalues > tol)


def pathway_quadratic_forms(F: jnp.ndarray, pathway_laplacian: jnp.ndarray) -> jnp.ndarray:
    """Return ``F[:, h]^T pathway_laplacian F[:, h]`` for every column ``h``."""
    return jnp.einsum("gh,gj,jh->h", F, pathway_laplacian, F)


def pathway_logprob(
    F: jnp.ndarray,
    pathway_laplacian: jnp.ndarray,
    rho: float,
    active: jnp.ndarray,
    *,
    rank: int | None = None,
) -> jnp.ndarray:
    """Evaluate the intrinsic Gaussian pathway prior using its rank.

    The pathway prior couples gene loadings only within *active* (slab)
    columns -- inactive (spike) columns carry no pathway term. The Laplacian
    may be rank deficient; its normalizing contribution uses the
    pseudo-determinant rank (see :func:`pathway_laplacian_rank`), while the
    quadratic form acts only through the supplied Laplacian.
    """
    eigenvalues = jnp.linalg.eigvalsh(pathway_laplacian)
    positive = eigenvalues > 1e-8
    inferred_rank = jnp.sum(positive)
    effective_rank = inferred_rank if rank is None else jnp.asarray(rank)
    log_pdet = jnp.sum(jnp.where(positive, jnp.log(jnp.maximum(eigenvalues, 1e-8)), 0.0))
    quadratic = pathway_quadratic_forms(F, pathway_laplacian)
    n_active = jnp.sum(active)
    normalizing = 0.5 * n_active * (effective_rank * jnp.log(rho) + log_pdet)
    kernel = -0.5 * rho * jnp.sum(jnp.where(active, quadratic, 0.0))
    return normalizing + kernel


def laplace_mrf_logprob(L: jnp.ndarray, graph: Graph, scale: float = 1.0, edge_weight: float = 1.0) -> jnp.ndarray:
    """Evaluate the edge-preserving Laplace MRF log prior for ``L``.

    The spec's prior is a *per-component* Laplace potential,
    ``exp(-gamma * sum_edges sum_k |L_{s,k} - L_{s',k}|)`` -- the Gaussian
    scale-mixture augmentation in the model spec (``diff_k | eta_k ~
    N(0, eta_k)``, ``eta_k ~ Exponential(gamma^2/2)``) marginalizes to
    exactly this elementwise L1 potential, independently per component ``k``.
    That is *not* the same distribution as an edge-wise Euclidean (L2) group
    norm over the ``k`` axis, so the sum below is taken over every
    ``(edge, k)`` pair, not of a per-edge norm.
    """
    differences = L[graph.senders] - L[graph.receivers]
    # Directed edges are intentional; callers can pass half-weight if using both directions.
    return -edge_weight * jnp.sum(jnp.abs(differences)) / scale


def log_pi_L(
    L: jnp.ndarray,
    X: jnp.ndarray,
    F: jnp.ndarray,
    alpha: jnp.ndarray,
    p_of_s: jnp.ndarray,
    graph: Graph,
    mrf_scale: float = 1.0,
    gaussian_scale: float = 1.0,
) -> jnp.ndarray:
    """Evaluate the unnormalized posterior block density for spatial factors."""
    return (
        poisson_log_likelihood(X, L, F, alpha, p_of_s)
        + standard_normal_logprob(L / gaussian_scale)
        - L.size * jnp.log(gaussian_scale)
        + laplace_mrf_logprob(L, graph, mrf_scale)
    )


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
    pathway_laplacian: jnp.ndarray | None = None,
    rho: float = 0.0,
) -> jnp.ndarray:
    """Evaluate the true conditional target for ``F`` in a joint chain."""
    active = csp.z == jnp.arange(csp.H)
    pathway = 0.0 if pathway_laplacian is None or rho <= 0.0 else pathway_logprob(F, pathway_laplacian, rho, active)
    return poisson_log_likelihood(X, L, F, alpha, p_of_s) + csp_log_prior_F(F, csp) + pathway


def log_pi(
    L: jnp.ndarray,
    F: jnp.ndarray,
    X: jnp.ndarray,
    alpha_p: jnp.ndarray,
    p_of_s: jnp.ndarray,
    graph: Graph,
    csp: CSPState,
    *,
    pathway_laplacian: jnp.ndarray,
    rho: float,
    gaussian_scale: float = 1.0,
    mrf_scale: float = 1.0,
) -> jnp.ndarray:
    """Evaluate the full posterior target independently of any proposal."""
    active = csp.z == jnp.arange(csp.H)
    return log_pi_L(L, X, F, alpha_p, p_of_s, graph, mrf_scale, gaussian_scale) + csp_log_prior_F(F, csp) + pathway_logprob(F, pathway_laplacian, rho, active)


def log_pi_block_L(
    L_full: jnp.ndarray,
    X: jnp.ndarray,
    F: jnp.ndarray,
    alpha: jnp.ndarray,
    p_of_s: jnp.ndarray,
    graph: Graph,
    mrf_scale: float = 1.0,
    gaussian_scale: float = 1.0,
) -> jnp.ndarray:
    """Evaluate the target used to score an ``L`` zone-blocked MH move.

    This is ``log_pi_L`` evaluated at the full field with the candidate
    block plugged in. The additive terms that involve only the fixed
    complement are identical for the current and proposed block and cancel
    in the Metropolis-Hastings ratio, so scoring the whole field is exactly
    equivalent to -- and much simpler to keep consistent with -- scoring the
    block's own full conditional.
    """
    return log_pi_L(L_full, X, F, alpha, p_of_s, graph, mrf_scale, gaussian_scale)


def log_pi_block_F(
    F_full: jnp.ndarray,
    X: jnp.ndarray,
    L: jnp.ndarray,
    alpha: jnp.ndarray,
    p_of_s: jnp.ndarray,
    csp: CSPState,
    pathway_laplacian: jnp.ndarray | None = None,
    rho: float = 0.0,
) -> jnp.ndarray:
    """Evaluate the target used to score an ``F`` zone-blocked MH move.

    Analogous to :func:`log_pi_block_L`: ``log_pi_F_joint`` at the full gene
    field, relying on the same complement-terms-cancel argument.
    """
    return log_pi_F_joint(F_full, X, L, alpha, p_of_s, csp, pathway_laplacian, rho)
