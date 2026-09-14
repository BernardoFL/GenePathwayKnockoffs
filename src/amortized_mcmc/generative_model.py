"""Full NumPyro generative model for prior simulation and reference MCMC."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def _stick_weights(v: jnp.ndarray) -> jnp.ndarray:
    """Convert stick fractions to finite categorical weights."""
    remaining = jnp.concatenate((jnp.ones((1,), dtype=v.dtype), jnp.cumprod(1.0 - v[:-1])))
    return v * remaining


def full_generative_model(
    n_spots: int,
    n_genes: int,
    H: int,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    n_patient_effects: int = 1,
    mrf_scale: float = 1.0,
    a_phi: float = 2.0,
    theta_infty: float = 1e4,
    e0: float = 2.0,
    f0: float = 1.0,
    c0: float = 2.0,
    d0: float = 1.0,
    a0: float = 2.0,
    b0: float = 1.0,
    pathway_laplacian: jnp.ndarray | None = None,
    a_rho: float = 1.0,
    b_rho: float = 1.0,
    X: jnp.ndarray | None = None,
):
    """Sample the complete §1/§7 hierarchy and Poisson observation model.

    ``z`` is a padded categorical vector: row ``h`` has support ``0..h``.
    The active column indicator is ``z_h == h``; inactive columns use the
    fixed spike precision ``theta_infty``.
    The model exposes scalar ``z_h`` sites so NumPyro's
    ``DiscreteHMCGibbs`` can enumerate and update each allocation while NUTS
    handles the continuous variables. Passing ``X`` conditions the model on
    observed counts for posterior reference-chain runs.

    When ``pathway_laplacian`` is supplied, active columns of ``F`` are drawn
    jointly per column from the proper Gaussian formed by *composing* the
    per-entry Horseshoe precision with the pathway coupling precision --
    ``F_{.,h} ~ N(0, (diag(phi_h / nu_{.,h}^2) + rho * pathway_laplacian)^{-1})``
    for active ``h``, and the ordinary per-entry Horseshoe conditional for
    inactive (spike) columns. This is the unique proper distribution
    proportional to the "per-entry Horseshoe times pathway-coupling kernel"
    product described in the model spec, so `Predictive` draws of `F`
    actually reflect the pathway prior instead of silently ignoring it (the
    diagonal Horseshoe term keeps the combined precision full rank even
    though the pathway Laplacian alone is rank deficient, so no
    pseudo-determinant convention is needed here -- that convention is used
    downstream only by ``rho``'s own Gibbs full conditional, which is
    independent of how ``F`` is *sampled* here since a constant that does
    not depend on ``F`` cancels in every Metropolis-Hastings ratio that
    holds ``phi``/``rho`` fixed). Omitting ``pathway_laplacian`` recovers the
    plain elementwise Horseshoe/CSP prior.
    """
    alpha_csp = numpyro.sample("alpha_csp", dist.Gamma(e0, f0))
    b_phi = numpyro.sample("b_phi", dist.Gamma(c0, d0))
    v = numpyro.sample(
        "v",
        dist.Beta(1.0, alpha_csp).expand((H - 1,)).to_event(1),
    )
    v_full = jnp.concatenate((v, jnp.ones((1,), dtype=v.dtype)))
    weights = _stick_weights(v_full)
    z_values = []
    active_prefix = jnp.asarray(True)
    for h in range(H):
        probs = weights[: h + 1]
        probs = probs / jnp.sum(probs)
        forced_spike = jnp.zeros((h + 1,), dtype=probs.dtype).at[0].set(1.0)
        probs = jnp.where(active_prefix, probs, forced_spike)
        z_h = numpyro.sample(f"z_{h}", dist.Categorical(probs=probs))
        z_values.append(z_h)
        active_prefix = active_prefix & (z_h == h)
    z = jnp.stack(z_values)
    active = z == jnp.arange(H)
    phi_draws = numpyro.sample(
        "phi_slab",
        dist.Gamma(a_phi, b_phi).expand((H,)).to_event(1),
    )
    phi = numpyro.deterministic("phi", jnp.where(active, phi_draws, theta_infty))
    lambda_local = numpyro.sample(
        "lambda_local",
        dist.HalfCauchy(1.0).expand((n_genes, H)).to_event(2),
    )
    if pathway_laplacian is None:
        F = numpyro.sample(
            "F",
            dist.Normal(0.0, lambda_local * jnp.sqrt(1.0 / phi)[None, :]).to_event(2),
        )
    else:
        rho = numpyro.sample("rho", dist.Gamma(a_rho, b_rho))
        diag_precision = phi[None, :] / (lambda_local * lambda_local)  # (n_genes, H)
        diagonal_terms = jax.vmap(jnp.diag, in_axes=1)(diag_precision)  # (H, n_genes, n_genes)
        pathway_terms = jnp.where(
            active[:, None, None], rho * pathway_laplacian[None, :, :], 0.0
        )
        precision = diagonal_terms + pathway_terms
        F_by_column = numpyro.sample(
            "F_by_column",
            dist.MultivariateNormal(loc=jnp.zeros((H, n_genes)), precision_matrix=precision),
        )
        F = numpyro.deterministic("F", F_by_column.T)
    L = numpyro.sample("L", dist.Normal(0.0, 1.0).expand((n_spots, H)).to_event(2))
    # Per-component L1 potential (see amortized_mcmc.target.laplace_mrf_logprob
    # for why this must not be an edge-wise L2 group norm over components).
    differences = L[senders] - L[receivers]
    numpyro.factor("laplace_mrf", -jnp.sum(jnp.abs(differences)) / mrf_scale)
    sigma_alpha_sq = numpyro.sample("sigma_alpha_sq", dist.InverseGamma(a0, b0))
    alpha_p = numpyro.sample(
        "alpha_p",
        dist.Normal(0.0, jnp.sqrt(sigma_alpha_sq)).expand((n_patient_effects,)).to_event(1),
    )
    p_of_s = jnp.zeros((n_spots,), dtype=jnp.int32)
    eta = L @ F.T + alpha_p[p_of_s, None]
    numpyro.sample("X", dist.Poisson(jnp.exp(eta)).to_event(2), obs=X)
