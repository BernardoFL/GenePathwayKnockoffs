"""Updates for patient-level Poisson log-intensity effects.

The patient effects, called ``alpha_p`` in the model specification, are kept
outside both neural proposal heads. Their Poisson-log-Gaussian conditional is
not conjugate, so this module uses exact univariate slice sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .sampler import ChainState


def _alpha_log_conditional(value, index, state: ChainState, sigma_alpha: float) -> jax.Array:
    """Return the unnormalized conditional log density for one ``alpha_p``.

    Boolean spot membership is implemented with masks rather than dynamic JAX
    indexing so this function remains compatible with ``jit`` and ``scan``.
    """
    eta_without_alpha = state.L @ state.F.T
    spots = state.p_of_s == index
    eta = eta_without_alpha + value
    masked_counts = jnp.where(spots[:, None], state.X, 0.0)
    masked_rate = jnp.where(spots[:, None], jnp.exp(eta), 0.0)
    return jnp.sum(masked_counts * eta - masked_rate) - 0.5 * (value / sigma_alpha) ** 2


def _slice_one(key, current, index, state, sigma_alpha, width, max_steps):
    """Draw one scalar with stepping-out slice sampling.

    Args:
        key: JAX PRNG key for all random choices in this update.
        current: Current value of the scalar effect.
        index: Patient-effect index being updated.
        state: Current latent chain state.
        sigma_alpha: Prior standard deviation.
        width: Initial slice-bracket width.
        max_steps: Maximum number of bracket expansion/shrink iterations.
    """
    key, y_key, left_key, right_key = jax.random.split(key, 4)
    log_y = _alpha_log_conditional(current, index, state, sigma_alpha) + jnp.log(jax.random.uniform(y_key))
    left = current - width * jax.random.uniform(left_key)
    right = left + width

    def expand_left(_, value):
        """Expand the bracket left while its endpoint remains on the slice."""
        left_value, right_value = value
        can_expand = _alpha_log_conditional(left_value, index, state, sigma_alpha) > log_y
        return (jnp.where(can_expand, left_value - width, left_value), right_value)

    def expand_right(_, value):
        """Expand the bracket right while its endpoint remains on the slice."""
        left_value, right_value = value
        can_expand = _alpha_log_conditional(right_value, index, state, sigma_alpha) > log_y
        return (left_value, jnp.where(can_expand, right_value + width, right_value))

    left, right = jax.lax.fori_loop(0, max_steps, expand_left, (left, right))
    left, right = jax.lax.fori_loop(0, max_steps, expand_right, (left, right))

    def shrink(_, value):
        """Propose within the bracket and shrink rejected portions."""
        draw_key, next_key, left_value, right_value, sample = value
        candidate = left_value + jax.random.uniform(draw_key) * (right_value - left_value)
        accepted = _alpha_log_conditional(candidate, index, state, sigma_alpha) >= log_y
        new_left = jnp.where(accepted, left_value, jnp.where(candidate < current, candidate, left_value))
        new_right = jnp.where(accepted, right_value, jnp.where(candidate >= current, candidate, right_value))
        new_sample = jnp.where(accepted, candidate, sample)
        draw_key, next_key = jax.random.split(next_key)
        return draw_key, next_key, new_left, new_right, new_sample

    initial = (right_key, key, left, right, current)
    return jax.lax.fori_loop(0, max_steps, shrink, initial)[-1]


def gibbs_step_alpha(
    key: jax.Array,
    state: ChainState,
    sigma_alpha: float = 1.0,
    width: float = 1.0,
    max_steps: int = 16,
) -> jax.Array:
    """Update every patient effect with exact univariate slice sampling.

    The returned vector is suitable for replacing ``ChainState.alpha``. No
    proposal-network parameters or CSP variables are consulted.
    """
    keys = jax.random.split(key, state.alpha.shape[0])
    alpha = state.alpha
    for index, index_key in enumerate(keys):
        alpha = alpha.at[index].set(
            _slice_one(index_key, alpha[index], index, state._replace(alpha=alpha), sigma_alpha, width, max_steps)
        )
    return alpha


# The model has three distinct "alpha"s -- the patient effect alpha_p, the
# CSP concentration alpha_CSP, and the MH acceptance probability alpha_MH.
# This alias spells the patient-effect update by its full model name so call
# sites never have to guess which "alpha" a bare `gibbs_step_alpha` means.
gibbs_step_alpha_p = gibbs_step_alpha


def gibbs_step_sigma_alpha_sq(key: jax.Array, alpha: jax.Array, a0: float = 2.0, b0: float = 1.0) -> jax.Array:
    """Draw ``sigma_alpha^2`` from its InverseGamma conjugate conditional.

    ``alpha_p ~ N(0, sigma_alpha^2)`` with prior ``sigma_alpha^2 ~
    InverseGamma(a0, b0)`` is the one part of the patient-effect block that
    *is* conjugate (unlike ``alpha_p`` itself, whose Poisson-log-Gaussian
    conditional needs the slice sampler above): the posterior is
    ``InverseGamma(a0 + P/2, b0 + 0.5 * sum(alpha_p^2))`` for ``P`` patients.
    """
    n_patients = alpha.shape[0]
    shape = a0 + 0.5 * n_patients
    rate = b0 + 0.5 * jnp.sum(alpha * alpha)
    return rate / jax.random.gamma(key, shape)
