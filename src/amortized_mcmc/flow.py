"""Numerical tools for conditional continuous normalizing flows.

The flow field is trained with conditional flow matching and sampled with a
fixed-step RK4 solver. Densities are evaluated by integrating the augmented
state containing the divergence correction; Hutchinson's estimator avoids
forming a full Jacobian for high-dimensional latent arrays.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def _rk4_step(field: Callable, time: jax.Array, value: jax.Array, step: jax.Array) -> jax.Array:
    """Advance one RK4 step for an ODE field at a signed time increment."""
    k1 = field(time, value)
    k2 = field(time + step / 2.0, value + step * k1 / 2.0)
    k3 = field(time + step / 2.0, value + step * k2 / 2.0)
    k4 = field(time + step, value + step * k3)
    return value + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_ode(field: Callable, initial: jax.Array, t0: float, t1: float, steps: int = 16) -> jax.Array:
    """Integrate an ODE with fixed-step fourth-order Runge-Kutta.

    The signed step permits the same function to integrate forward for
    sampling or backward for density evaluation. ``field`` must accept
    ``(time, state)`` and return an array with the state's shape.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    step = (t1 - t0) / steps

    def body(index, value):
        """Compute one loop-carried RK4 update."""
        time = t0 + index * step
        return _rk4_step(field, time, value, step)

    return jax.lax.fori_loop(0, steps, body, initial)


def divergence_hutchinson(field: Callable, time: jax.Array, value: jax.Array, key: jax.Array) -> jax.Array:
    """Estimate ``trace(d field / d value)`` with a Rademacher probe vector."""
    noise = jax.random.rademacher(key, value.shape, dtype=value.dtype)
    _, jvp = jax.jvp(lambda state: field(time, state), (value,), (noise,))
    return jnp.sum(noise * jvp)


def integrate_log_density(
    field: Callable,
    value: jax.Array,
    base_logpdf: Callable[[jax.Array], jax.Array],
    key: jax.Array,
    t0: float = 0.0,
    t1: float = 1.0,
    steps: int = 16,
) -> jax.Array:
    """Evaluate a flow density by integrating the augmented state.

    Integrating from the requested value at t=1 back to the base at t=0 gives
    ``log q(value) = log p_base(value_0) - integral(div v dt)``.
    Args:
        field: Conditional velocity field ``(time, state) -> velocity``.
        value: Point at which the transformed density is requested.
        base_logpdf: Log density of the base distribution.
        key: PRNG key for the Hutchinson trace probe.
        t0: Base-distribution time, normally zero.
        t1: Data time, normally one.
        steps: Number of fixed RK4 steps.
    """
    flat_value = value.reshape(-1)
    keys = jax.random.split(key, steps)

    def augmented_field(time, augmented):
        """Evaluate state velocity and its estimated divergence together."""
        state = augmented[:-1].reshape(value.shape)
        velocity = field(time, state)
        divergence = divergence_hutchinson(field, time, state, keys[0])
        return jnp.concatenate((velocity.reshape(-1), jnp.asarray([divergence], dtype=value.dtype)))

    initial = jnp.concatenate((flat_value, jnp.zeros((1,), dtype=value.dtype)))
    integrated = integrate_ode(augmented_field, initial, t1, t0, steps)
    base_value = integrated[:-1].reshape(value.shape)
    divergence_integral = integrated[-1]
    return base_logpdf(base_value) + divergence_integral


def standard_normal_logpdf(value: jax.Array) -> jax.Array:
    """Return the summed standard-normal log density for an array."""
    return -0.5 * jnp.sum(value * value + jnp.log(2.0 * jnp.pi))


def conditional_flow_matching_loss(
    field: Callable,
    key: jax.Array,
    base: jax.Array,
    target: jax.Array,
    *field_args,
) -> jax.Array:
    """Compute straight-line conditional flow-matching loss.

    A base sample and a paired target transition define a random interpolation
    time. The field is regressed against the constant velocity of that line.
    Additional ``field_args`` are forwarded to the supplied field callable.
    """
    time_key, noise_key = jax.random.split(key)
    time = jax.random.uniform(time_key)
    interpolation = (1.0 - time) * base + time * target
    target_velocity = target - base
    # Small noise avoids a degenerate training path when simulated pairs repeat.
    interpolation = interpolation + 1e-5 * jax.random.normal(noise_key, interpolation.shape)
    predicted = field(time, interpolation, *field_args)
    return jnp.mean((predicted - target_velocity) ** 2)