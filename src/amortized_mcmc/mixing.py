"""Utilities for comparing MCMC traces and estimating effective sample size."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp



def _autocorrelation(values: jax.Array, lag: int) -> jax.Array:
    """Compute the normalized autocorrelation of a scalar trace at one lag."""
    centered = values - jnp.mean(values)
    numerator = jnp.mean(centered[:-lag] * centered[lag:]) if lag else jnp.mean(centered * centered)
    denominator = jnp.mean(centered * centered) + 1e-8
    return numerator / denominator


def effective_sample_size(values: jax.Array, max_lag: int | None = None) -> jax.Array:
    """Estimate scalar-trace ESS with a positive autocorrelation sequence."""
    values = jnp.asarray(values).reshape(-1)
    if values.shape[0] < 4:
        return jnp.asarray(values.shape[0], dtype=jnp.float32)
    max_lag = min(values.shape[0] // 2, max_lag or values.shape[0] // 2)
    correlations = jnp.asarray([_autocorrelation(values, lag) for lag in range(1, max_lag)])
    positive = jnp.cumprod((correlations > 0).astype(jnp.int32)).astype(bool)
    integrated = 1.0 + 2.0 * jnp.sum(jnp.where(positive, correlations, 0.0))
    return values.shape[0] / jnp.maximum(integrated, 1.0)


def compare_mixing(
    key: jax.Array,
    initial_state,
    mh_step_fn: Callable,
    gibbs_step_fn: Callable,
    observable: Callable,
    n_steps: int,
) -> dict[str, jax.Array]:
    """Run comparable traces for amortized MH and a plain Gibbs baseline.

    Step functions receive ``(key, state)`` and return a new state. The helper
    intentionally treats both kernels as black boxes so a project can provide
    its reference Gibbs implementation without coupling it to the proposal.
    The two step functions must accept ``(key, state)`` and return a state
    with the same pytree structure. ``observable`` maps each state to the
    scalar or array trace whose mixing should be compared.
    """
    mh_keys, gibbs_keys = jax.random.split(key, 2)
    mh_keys = jax.random.split(mh_keys, n_steps)
    gibbs_keys = jax.random.split(gibbs_keys, n_steps)

    def run(step_fn, keys):
        """Scan one kernel over its independent sequence of PRNG keys."""
        def body(state, step_key):
            """Advance the chain once and emit the requested observable."""
            next_state = step_fn(step_key, state)
            return next_state, observable(next_state)

        _, trace = jax.lax.scan(body, initial_state, keys)
        return trace

    mh_trace = run(mh_step_fn, mh_keys)
    gibbs_trace = run(gibbs_step_fn, gibbs_keys)
    return {
        "mh_trace": mh_trace,
        "gibbs_trace": gibbs_trace,
        "mh_ess": effective_sample_size(mh_trace),
        "gibbs_ess": effective_sample_size(gibbs_trace),
    }
