"""Cumulative shrinkage-process prior state and auxiliary updates."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CSPState(NamedTuple):
    """Finite-``H`` CSP variables and fixed hyperparameters."""
    v: jax.Array
    z: jax.Array
    phi: jax.Array
    concentration: jax.Array
    b_phi: jax.Array
    lam: jax.Array
    a_phi: float
    theta_infty: float

    @property
    def H(self) -> int:
        """Return the configured CSP truncation width."""
        return int(self.phi.shape[0])


def _stick_weights(v: jax.Array) -> jax.Array:
    """Convert finite stick-breaking fractions into category weights."""
    remaining = jnp.concatenate((jnp.ones((1,), dtype=v.dtype), jnp.cumprod(1.0 - v[:-1])))
    return v * remaining


def sample_csp_prior(
    key: jax.Array,
    H: int,
    n_genes: int,
    *,
    concentration: float = 1.0,
    a_phi: float = 2.0,
    b_phi: float = 1.0,
    theta_infty: float = 1e4,
) -> CSPState:
    """Draw CSP sticks, ordered allocations, precisions, and local scales."""
    if H < 1:
        raise ValueError("H must be positive")
    v_key, z_key, phi_key, lam_key = jax.random.split(key, 4)
    v = jnp.concatenate(
        (jax.random.beta(v_key, 1.0, concentration, (H - 1,)), jnp.ones((1,))),
    )
    weights = _stick_weights(v)
    z_keys = jax.random.split(z_key, H)
    z_values = []
    active_prefix = jnp.asarray(True)
    for index, subkey in enumerate(z_keys):
        draw = jax.random.categorical(subkey, jnp.log(weights[: index + 1]))
        draw = jnp.where(active_prefix, draw, 0)
        z_values.append(draw)
        active_prefix = active_prefix & (draw == index)
    z = jnp.asarray(z_values, dtype=jnp.int32)
    active = z == jnp.arange(H)
    phi_draws = jax.random.gamma(phi_key, a_phi, (H,)) / b_phi
    phi = jnp.where(active, phi_draws, theta_infty)
    lam = 1.0 / jax.random.gamma(lam_key, 0.5, (n_genes, H))
    return CSPState(v, z, phi, jnp.asarray(concentration), jnp.asarray(b_phi), lam, a_phi, theta_infty)


def _column_log_likelihood(column, lam_column, precision):
    """Compute the Gaussian log-kernel for one loading column."""
    variance_precision = precision / (lam_column * lam_column)
    return 0.5 * jnp.sum(jnp.log(variance_precision) - variance_precision * column * column)


def _sample_z(key, F, csp: CSPState) -> jax.Array:
    """Draw ordered CSP allocation indicators conditional on loadings."""
    weights = _stick_weights(csp.v)
    keys = jax.random.split(key, csp.H)
    result = []
    active_prefix = jnp.asarray(True)
    for h, subkey in enumerate(keys):
        candidates = jnp.arange(h + 1)
        candidate_phi = jnp.where(csp.z[: h + 1] == candidates, csp.phi[: h + 1], csp.theta_infty)
        log_probs = jnp.log(weights[: h + 1] + 1e-30) + jax.vmap(
            lambda precision: _column_log_likelihood(F[:, h], csp.lam[:, h], precision)
        )(candidate_phi)
        draw = jax.random.categorical(subkey, log_probs)
        draw = jnp.where(active_prefix, draw, 0)
        result.append(draw)
        active_prefix = active_prefix & (draw == h)
    return jnp.asarray(result, dtype=jnp.int32)


def _log_concentration(value, v, e0, f0):
    """Evaluate the unnormalized finite-stick concentration conditional."""
    return (e0 + v.shape[0] - 1.0) * jnp.log(value) - value * (f0 - jnp.sum(jnp.log1p(-v[:-1])))


def _slice_concentration(key, current, v, e0, f0, width=1.0, steps=16):
    """Draw CSP concentration with stepping-out slice sampling."""
    draw_key, left_key, right_key = jax.random.split(key, 3)
    log_level = _log_concentration(current, v, e0, f0) + jnp.log(jax.random.uniform(draw_key))
    left = jnp.maximum(1e-6, current - width * jax.random.uniform(left_key))
    right = left + width

    def expand_left(_, interval):
        """Expand the concentration bracket toward smaller values."""
        lower, upper = interval
        can_expand = _log_concentration(lower, v, e0, f0) > log_level
        return jnp.where(can_expand, jnp.maximum(1e-6, lower - width), lower), upper

    def expand_right(_, interval):
        """Expand the concentration bracket toward larger values."""
        lower, upper = interval
        can_expand = _log_concentration(upper, v, e0, f0) > log_level
        return lower, jnp.where(can_expand, upper + width, upper)

    left, right = jax.lax.fori_loop(0, steps, expand_left, (left, right))
    left, right = jax.lax.fori_loop(0, steps, expand_right, (left, right))

    def shrink(index, interval):
        """Shrink rejected concentration bracket regions."""
        lower, upper, sample = interval
        candidate = lower + jax.random.uniform(jax.random.fold_in(key, index)) * (upper - lower)
        accepted = _log_concentration(candidate, v, e0, f0) >= log_level
        lower = jnp.where(accepted, lower, jnp.where(candidate < current, candidate, lower))
        upper = jnp.where(accepted, upper, jnp.where(candidate >= current, candidate, upper))
        sample = jnp.where(accepted, candidate, sample)
        return lower, upper, sample

    return jax.lax.fori_loop(0, steps, shrink, (left, right, current))[-1]


def _sample_v(key, z, concentration):
    """Draw conjugate stick fractions from allocation counts."""
    H = z.shape[0]
    keys = jax.random.split(key, max(H - 1, 1))
    draws = []
    for h, subkey in enumerate(keys[: H - 1]):
        successes = jnp.sum(z == h)
        failures = jnp.sum(z > h)
        draws.append(jax.random.beta(subkey, 1.0 + successes, concentration + failures))
    return jnp.concatenate((jnp.asarray(draws), jnp.ones((1,))))


def _sample_phi(key, F, csp: CSPState, z):
    """Draw active column precisions and restore fixed spike precisions."""
    keys = jax.random.split(key, csp.H)
    active = z == jnp.arange(csp.H)
    shape = csp.a_phi + 0.5 * F.shape[0]
    rate = csp.b_phi + 0.5 * jnp.sum((F / csp.lam) ** 2, axis=0)
    draws = jax.random.gamma(key, shape, (csp.H,)) / rate
    return jnp.where(active, draws, csp.theta_infty)


def gibbs_step_csp(
    key: jax.Array,
    F: jax.Array,
    csp: CSPState | None = None,
    *,
    e0: float = 2.0,
    f0: float = 1.0,
    c0: float = 2.0,
    d0: float = 1.0,
) -> CSPState:
    """Run one CSP auxiliary update outside all neural proposal heads."""
    chain_state = None
    if csp is None:
        chain_state = F
        F = chain_state.F
        csp = chain_state.csp
        if csp is None:
            raise ValueError("ChainState.csp must be initialized before a CSP update")
    z_key, v_key, phi_key, b_key, alpha_key = jax.random.split(key, 5)
    z = _sample_z(z_key, F, csp)
    v = _sample_v(v_key, z, csp.concentration)
    phi = _sample_phi(phi_key, F, csp, z)
    active = z == jnp.arange(csp.H)
    active_count = jnp.sum(active)
    b_phi = jax.random.gamma(b_key, c0 + csp.a_phi * active_count) / (
        d0 + jnp.sum(jnp.where(active, phi, 0.0))
    )
    concentration = _slice_concentration(alpha_key, csp.concentration, v, e0, f0)
    updated = csp._replace(v=v, z=z, phi=phi, concentration=concentration, b_phi=b_phi)
    return updated if chain_state is None else chain_state._replace(csp=updated)
