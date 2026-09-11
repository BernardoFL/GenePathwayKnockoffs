"""Full NumPyro generative model for prior simulation and reference MCMC."""

from __future__ import annotations

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
    F = numpyro.sample(
        "F",
        dist.Normal(0.0, lambda_local * jnp.sqrt(1.0 / phi)[None, :]).to_event(2),
    )
    L = numpyro.sample("L", dist.Normal(0.0, 1.0).expand((n_spots, H)).to_event(2))
    differences = L[senders] - L[receivers]
    numpyro.factor(
        "laplace_mrf",
        -jnp.sum(jnp.sqrt(jnp.sum(differences * differences, axis=-1) + 1e-8)) / mrf_scale,
    )
    alpha_p = numpyro.sample(
        "alpha_p",
        dist.Normal(0.0, 1.0).expand((n_patient_effects,)).to_event(1),
    )
    p_of_s = jnp.zeros((n_spots,), dtype=jnp.int32)
    eta = L @ F.T + alpha_p[p_of_s, None]
    numpyro.sample("X", dist.Poisson(jnp.exp(eta)).to_event(2), obs=X)
