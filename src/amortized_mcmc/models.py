"""Equinox amortizer heads for spatial factors and gene loadings.

The spatial head is a graph-attention conditional vector field. The loading
head is graph-free because genes have no spot topology. Both heads use the
same conditional-flow interface so their proposals can be sampled and scored
in both MH directions.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .graph import Graph
from .flow import integrate_log_density, integrate_ode, standard_normal_logpdf


def _normal_logpdf(value: jax.Array, mean: jax.Array, log_scale: jax.Array) -> jax.Array:
    """Return the summed diagonal-normal log density."""
    z = (value - mean) * jnp.exp(-log_scale)
    return jnp.sum(-0.5 * (z * z + 2.0 * log_scale + jnp.log(2.0 * jnp.pi)))


def _edge_softmax(logits: jax.Array, receivers: jax.Array, n_nodes: int) -> jax.Array:
    """Normalize edge logits independently over incoming edges per node."""
    maxima = jnp.full((n_nodes,), -jnp.inf).at[receivers].max(logits)
    exponentials = jnp.exp(logits - maxima[receivers])
    normalizers = jnp.zeros((n_nodes,)).at[receivers].add(exponentials)
    return exponentials / normalizers[receivers]


class GATLayer(eqx.Module):
    """Single graph-attention message-passing layer."""
    projection: eqx.nn.Linear
    attention: eqx.nn.Linear
    negative_slope: float = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        """Initialize projection and pairwise attention weights."""
        projection_key, attention_key = jax.random.split(key)
        self.projection = eqx.nn.Linear(in_size, out_size, key=projection_key)
        self.attention = eqx.nn.Linear(2 * out_size, 1, key=attention_key)
        self.negative_slope = 0.2

    def __call__(self, features: jax.Array, graph: Graph) -> jax.Array:
        """Aggregate sender features into receiver nodes with attention."""
        projected = jax.vmap(self.projection)(features)
        pair_features = jnp.concatenate(
            (projected[graph.senders], projected[graph.receivers]), axis=-1
        )
        logits = jax.vmap(self.attention)(pair_features).squeeze(-1)
        logits = jax.nn.leaky_relu(logits, negative_slope=self.negative_slope)
        weights = _edge_softmax(logits, graph.receivers, features.shape[0])
        aggregated = jnp.zeros_like(projected).at[graph.receivers].add(
            weights[:, None] * projected[graph.senders]
        )
        return jax.nn.elu(aggregated)


class SpatialProposal(eqx.Module):
    """Fixed-``H`` GAT conditional flow for spatial factors ``L``."""
    gat: tuple[GATLayer, ...]
    velocity: eqx.nn.Linear
    H: int = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)
    coordinate_dim: int = eqx.field(static=True)

    def __init__(self, H: int, observation_dim: int, hidden_dim: int = 32, depth: int = 2, *, key: jax.Array, coordinate_dim: int = 2):
        """Initialize a coordinate-aware GAT velocity field."""
        if depth < 1:
            raise ValueError("depth must be positive")
        keys = jax.random.split(key, depth + 1)
        layers = []
        in_size = H + observation_dim + coordinate_dim
        for index in range(depth):
            layers.append(GATLayer(in_size, hidden_dim, keys[index]))
            in_size = hidden_dim
        self.gat = tuple(layers)
        self.velocity = eqx.nn.Linear(hidden_dim + H + 1, H, key=keys[-1])
        self.H = H
        self.observation_dim = observation_dim
        self.coordinate_dim = coordinate_dim

    def flow_field(self, t: jax.Array, L_t: jax.Array, X: jax.Array, L: jax.Array, graph: Graph, coordinates: jax.Array | None = None) -> jax.Array:
        """Evaluate ``v_phi(t, L_t | X, L)`` at every spot."""
        if coordinates is None:
            coordinates = jnp.zeros((L_t.shape[0], self.coordinate_dim), dtype=L_t.dtype)
        normalized_X = jnp.log1p(X)
        normalized_X = normalized_X / (jnp.mean(normalized_X, axis=0, keepdims=True) + 1e-6)
        features = jnp.concatenate((L, normalized_X, coordinates), axis=-1)
        for layer in self.gat:
            features = layer(features, graph)
        time_features = jnp.broadcast_to(t, (L_t.shape[0], 1))
        return jax.vmap(self.velocity)(jnp.concatenate((features, L_t, time_features), axis=-1))

    def flow_field_L(self, t: jax.Array, L_t: jax.Array, X: jax.Array, L: jax.Array, graph: Graph, coordinates: jax.Array | None = None) -> jax.Array:
        """Alias for the spatial flow-field interface used by training code."""
        return self.flow_field(t, L_t, X, L, graph, coordinates)

    def parameters(self, t: jax.Array, L_t: jax.Array, X: jax.Array, L: jax.Array, graph: Graph, coordinates: jax.Array | None = None) -> jax.Array:
        """Backward-compatible alias returning the spatial velocity field."""
        return self.flow_field(t, L_t, X, L, graph, coordinates)

    def log_q(
        self,
        value: jax.Array,
        X: jax.Array,
        current: jax.Array,
        graph: Graph,
        coordinates: jax.Array | None = None,
        *,
        key: jax.Array | None = None,
        steps: int = 16,
    ) -> jax.Array:
        """Evaluate the flow-induced density at a proposed spatial state."""
        density_key = jax.random.PRNGKey(0) if key is None else key
        field = lambda time, state: self.flow_field(time, state, X, current, graph, coordinates)
        return integrate_log_density(field, value, standard_normal_logpdf, density_key, steps=steps)

    def sample_and_log_q(
        self,
        key: jax.Array,
        X: jax.Array,
        current: jax.Array,
        graph: Graph,
        coordinates: jax.Array | None = None,
        *,
        steps: int = 16,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample an L proposal from the base flow and score it."""
        sample_key, density_key = jax.random.split(key)
        base = jax.random.normal(sample_key, current.shape)
        field = lambda time, state: self.flow_field(time, state, X, current, graph, coordinates)
        proposal = integrate_ode(field, base, 0.0, 1.0, steps)
        return proposal, self.log_q(proposal, X, current, graph, key=density_key, steps=steps)


class LoadingProposal(eqx.Module):
    """Independent graph-free MLP conditional flow for gene loadings ``F``."""
    mlp: eqx.nn.MLP
    loading_shape: tuple[int, int] = eqx.field(static=True)

    def __init__(self, observation_dim: int, n_genes: int, latent_dim: int, hidden_dim: int = 128, depth: int = 2, *, key: jax.Array):
        """Initialize the fixed gene-panel and CSP-width loading field."""
        size = n_genes * latent_dim
        self.mlp = eqx.nn.MLP(observation_dim + size + 1, size, hidden_dim, depth, key=key)
        self.loading_shape = (n_genes, latent_dim)

    def flow_field(self, t: jax.Array, F_t: jax.Array, X: jax.Array, F: jax.Array) -> jax.Array:
        """Evaluate the loading velocity conditioned on X and current F."""
        context = jnp.mean(jnp.log1p(X), axis=0)
        flat_current = F.reshape(-1)
        flat_value = F_t.reshape(-1)
        output = self.mlp(jnp.concatenate((context, flat_current, jnp.asarray([t]))))
        return output.reshape(self.loading_shape)

    def flow_field_F(self, t: jax.Array, F_t: jax.Array, X: jax.Array, F: jax.Array) -> jax.Array:
        """Alias for the loading flow-field training interface."""
        return self.flow_field(t, F_t, X, F)

    def log_q(
        self,
        value: jax.Array,
        X: jax.Array,
        current: jax.Array,
        *,
        key: jax.Array | None = None,
        steps: int = 16,
    ) -> jax.Array:
        """Evaluate the flow-induced density at a proposed loading state."""
        density_key = jax.random.PRNGKey(0) if key is None else key
        field = lambda time, state: self.flow_field(time, state, X, current)
        return integrate_log_density(field, value, standard_normal_logpdf, density_key, steps=steps)

    def sample_and_log_q(self, key: jax.Array, X: jax.Array, current: jax.Array, *, steps: int = 16) -> tuple[jax.Array, jax.Array]:
        """Sample and score a loading proposal with the conditional flow."""
        sample_key, density_key = jax.random.split(key)
        base = jax.random.normal(sample_key, current.shape)
        field = lambda time, state: self.flow_field(time, state, X, current)
        proposal = integrate_ode(field, base, 0.0, 1.0, steps)
        return proposal, self.log_q(proposal, X, current, key=density_key, steps=steps)


class AmortizedProposal(NamedTuple):
    """Pair of independent spatial and loading proposal heads."""
    spatial: SpatialProposal
    loading: LoadingProposal
