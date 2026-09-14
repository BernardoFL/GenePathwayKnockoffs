"""Exact, deterministic coupling-flow primitives for amortized proposals.

The proposal density is evaluated by inverting affine coupling layers and
summing their diagonal log-Jacobians. There is no ODE solver, no trace
estimator, and no randomness anywhere in a density evaluation: a stochastic
``log q`` would make the Metropolis-Hastings acceptance ratio noisy and break
the exactness the FDR guarantee depends on, so that machinery is deliberately
absent from this module (see the project's non-goals).

Two coupling-flow shapes live here:

* :class:`AffineCouplingLayer` / :class:`CouplingFlow` transform a single flat
  vector with a fixed dimension. Useful for standalone unit tests of the
  primitive itself.
* :class:`GraphCouplingLayer` transforms a per-node array indexed by an
  arbitrary graph, with a boolean passive/active mask over *nodes* (not over
  vector coordinates). A caller-supplied stack of message-passing modules
  (e.g. GAT layers, injected rather than imported here to keep this module
  graph-network-agnostic) conditions each active node's affine parameters on
  the passive nodes' values plus per-node context. Because the conditioner is
  message passing over a graph, this layer is agnostic to the node count, so
  the resulting flow generalizes across spot counts / gene panels of any
  size, unlike the flat vector form.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def standard_normal_logpdf(value: jax.Array) -> jax.Array:
    """Return the summed standard-normal log density for an array."""
    return -0.5 * jnp.sum(value * value + jnp.log(2.0 * jnp.pi))


def student_t_logpdf(value: jax.Array, degrees_of_freedom: float = 4.0) -> jax.Array:
    """Return the summed standard Student-t log density for an array."""
    nu = jnp.asarray(degrees_of_freedom, dtype=value.dtype)
    normalizer = jax.scipy.special.gammaln((nu + 1.0) / 2.0)
    normalizer -= jax.scipy.special.gammaln(nu / 2.0)
    normalizer -= 0.5 * (jnp.log(nu) + jnp.log(jnp.pi))
    return jnp.sum(normalizer - 0.5 * (nu + 1.0) * jnp.log1p(value * value / nu))


def masked_normal_logpdf(value: jax.Array, row_mask: jax.Array) -> jax.Array:
    """Sum standard-normal log density over rows selected by ``row_mask``.

    ``value`` has shape ``(n_nodes, dim)``; rows where ``row_mask`` is
    ``False`` (e.g. always-passive context nodes that were never sampled)
    contribute nothing, so this is the correct base density for a proposal
    that only draws a subset of nodes.
    """
    per_row = -0.5 * jnp.sum(value * value + jnp.log(2.0 * jnp.pi), axis=-1)
    return jnp.sum(jnp.where(row_mask, per_row, 0.0))


def masked_student_t_logpdf(value: jax.Array, row_mask: jax.Array, degrees_of_freedom: float = 4.0) -> jax.Array:
    """Sum standard Student-t log density over rows selected by ``row_mask``."""
    nu = jnp.asarray(degrees_of_freedom, dtype=value.dtype)
    normalizer = jax.scipy.special.gammaln((nu + 1.0) / 2.0)
    normalizer -= jax.scipy.special.gammaln(nu / 2.0)
    normalizer -= 0.5 * (jnp.log(nu) + jnp.log(jnp.pi))
    per_row = jnp.sum(normalizer - 0.5 * (nu + 1.0) * jnp.log1p(value * value / nu), axis=-1)
    return jnp.sum(jnp.where(row_mask, per_row, 0.0))


class AffineCouplingLayer(eqx.Module):
    """One conditional affine coupling layer with a fixed binary mask."""

    conditioner: eqx.nn.MLP
    mask: jax.Array
    scale_clip: float = eqx.field(static=True)

    def __init__(self, dimension: int, context_dim: int, mask: jax.Array, hidden_dim: int, key: jax.Array, scale_clip: float = 5.0):
        """Initialize a masked affine transform."""
        transformed_dim = int(jnp.sum(1 - mask))
        passive_dim = int(jnp.sum(mask))
        self.conditioner = eqx.nn.MLP(
            passive_dim + context_dim,
            2 * transformed_dim,
            hidden_dim,
            2,
            key=key,
        )
        self.mask = mask
        self.scale_clip = scale_clip

    def _parameters(self, passive: jax.Array, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Produce bounded scale and shift parameters for active coordinates."""
        parameters = self.conditioner(jnp.concatenate((passive, context)))
        scale, shift = jnp.split(parameters, 2)
        return self.scale_clip * jnp.tanh(scale / self.scale_clip), shift

    def forward(self, value: jax.Array, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Transform a base value to data space and return its log-det."""
        passive = value[self.mask]
        scale, shift = self._parameters(passive, context)
        active = value[~self.mask] * jnp.exp(scale) + shift
        result = value.at[~self.mask].set(active)
        return result, jnp.sum(scale)

    def inverse(self, value: jax.Array, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Invert the layer and return the inverse log-det."""
        passive = value[self.mask]
        scale, shift = self._parameters(passive, context)
        base_active = (value[~self.mask] - shift) * jnp.exp(-scale)
        result = value.at[~self.mask].set(base_active)
        return result, -jnp.sum(scale)


class CouplingFlow(eqx.Module):
    """Stacked conditional affine coupling flow with an exact density."""

    layers: tuple[AffineCouplingLayer, ...]
    dimension: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    base: str = eqx.field(static=True)
    degrees_of_freedom: float = eqx.field(static=True)

    def __init__(self, dimension: int, context_dim: int, *, hidden_dim: int = 64, depth: int = 4, key: jax.Array, base: str = "normal", degrees_of_freedom: float = 4.0):
        """Initialize alternating affine coupling masks."""
        if dimension < 2 or context_dim < 0 or depth < 1:
            raise ValueError("dimension must be >= 2, context_dim nonnegative, and depth positive")
        if base not in {"normal", "student_t"}:
            raise ValueError("base must be 'normal' or 'student_t'")
        keys = jax.random.split(key, depth)
        layers = []
        for index, layer_key in enumerate(keys):
            mask = (jnp.arange(dimension) + index) % 2 == 0
            layers.append(AffineCouplingLayer(dimension, context_dim, mask, hidden_dim, layer_key))
        self.layers = tuple(layers)
        self.dimension = dimension
        self.context_dim = context_dim
        self.base = base
        self.degrees_of_freedom = degrees_of_freedom

    def _base_logpdf(self, value: jax.Array) -> jax.Array:
        """Evaluate the configured exact base density."""
        if self.base == "normal":
            return standard_normal_logpdf(value)
        return student_t_logpdf(value, self.degrees_of_freedom)

    def forward(self, base_value: jax.Array, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Map a base value to data space and return the forward log-det."""
        value = base_value.reshape(-1)
        log_det = jnp.asarray(0.0, dtype=value.dtype)
        for layer in self.layers:
            value, layer_log_det = layer.forward(value, context)
            log_det = log_det + layer_log_det
        return value.reshape(base_value.shape), log_det

    def inverse(self, value: jax.Array, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Map a data value to base space and return the inverse log-det."""
        base_value = value.reshape(-1)
        log_det = jnp.asarray(0.0, dtype=base_value.dtype)
        for layer in self.layers[::-1]:
            base_value, layer_log_det = layer.inverse(base_value, context)
            log_det = log_det + layer_log_det
        return base_value.reshape(value.shape), log_det

    def logdensity(self, value: jax.Array, context: jax.Array) -> jax.Array:
        """Evaluate ``log q(value | context)`` deterministically."""
        base_value, inverse_log_det = self.inverse(value, context)
        return self._base_logpdf(base_value) + inverse_log_det

    def sample_and_logdensity(self, key: jax.Array, context: jax.Array, shape: tuple[int, ...] | None = None) -> tuple[jax.Array, jax.Array]:
        """Sample once from the base and return the exact proposal density."""
        sample_shape = (self.dimension,) if shape is None else shape
        if int(jnp.prod(jnp.asarray(sample_shape))) != self.dimension:
            raise ValueError("sample shape must contain exactly `dimension` values")
        if self.base == "normal":
            base_value = jax.random.normal(key, sample_shape)
        else:
            base_value = jax.random.t(key, self.degrees_of_freedom, sample_shape)
        value, log_det = self.forward(base_value, context)
        return value, self._base_logpdf(base_value) - log_det


class GraphCouplingLayer(eqx.Module):
    """One affine coupling layer masked over graph nodes.

    Unlike :class:`AffineCouplingLayer`, the passive/active split is a
    boolean mask over *nodes* of a graph, supplied fresh at every call
    (so the same layer can serve different zone blocks or the empty-context
    global jump). A caller-injected stack of message-passing modules (each
    callable as ``layer(node_features, graph) -> node_features``, the
    interface :class:`~amortized_mcmc.models.GATLayer` satisfies) conditions
    every active node's affine parameters on the passive nodes' values and a
    fixed per-node context. Active nodes are masked to zero in the
    conditioner's input so they cannot inform their own transform or each
    other's within a layer -- exactly the independence structure a coupling
    layer requires for its log-det to stay a diagonal sum. Because message
    passing has no fixed node count, this layer is agnostic to the size of
    the graph it is applied to.
    """

    message_passing: tuple[eqx.Module, ...]
    readout: eqx.nn.Linear
    scale_clip: float = eqx.field(static=True)

    def __init__(self, message_passing, hidden_dim: int, node_dim: int, key: jax.Array, scale_clip: float = 5.0):
        """Store the conditioner stack and its node-wise affine readout."""
        self.message_passing = tuple(message_passing)
        self.readout = eqx.nn.Linear(hidden_dim, 2 * node_dim, key=key)
        self.scale_clip = scale_clip

    def _node_params(self, value: jax.Array, passive: jax.Array, node_context: jax.Array, graph) -> tuple[jax.Array, jax.Array]:
        """Run the conditioner and return bounded per-node scale and shift."""
        masked_value = jnp.where(passive[:, None], value, 0.0)
        features = jnp.concatenate((masked_value, node_context, passive[:, None].astype(value.dtype)), axis=-1)
        for layer in self.message_passing:
            features = layer(features, graph)
        scale, shift = jnp.split(jax.vmap(self.readout)(features), 2, axis=-1)
        return self.scale_clip * jnp.tanh(scale / self.scale_clip), shift

    def forward(self, value: jax.Array, passive: jax.Array, node_context: jax.Array, graph) -> tuple[jax.Array, jax.Array]:
        """Transform active nodes to data space and return the summed log-det."""
        scale, shift = self._node_params(value, passive, node_context, graph)
        active = ~passive
        new_value = jnp.where(active[:, None], value * jnp.exp(scale) + shift, value)
        log_det = jnp.sum(jnp.where(active, jnp.sum(scale, axis=-1), 0.0))
        return new_value, log_det

    def inverse(self, value: jax.Array, passive: jax.Array, node_context: jax.Array, graph) -> tuple[jax.Array, jax.Array]:
        """Invert active nodes to base space and return the inverse log-det."""
        scale, shift = self._node_params(value, passive, node_context, graph)
        active = ~passive
        base_value = jnp.where(active[:, None], (value - shift) * jnp.exp(-scale), value)
        log_det = -jnp.sum(jnp.where(active, jnp.sum(scale, axis=-1), 0.0))
        return base_value, log_det