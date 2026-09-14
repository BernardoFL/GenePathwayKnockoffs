"""Equinox amortizer heads: GAT-conditioned coupling flows over graphs.

Both heads are structurally parallel conditional coupling normalizing flows
(``GraphCouplingLayer`` stacks from :mod:`amortized_mcmc.flow`) whose
conditioner is a Graph Attention Network keyed to the head's own index-set
graph -- the spot graph for :class:`SpatialProposal` (Head 1), the pathway
graph for :class:`LoadingProposal` (Head 2). Neither head is tied to a fixed
node count: GAT message passing and the per-node coupling readout both
operate independently of how many spots or genes are present.

Every proposal is an *independence* proposal: a fresh draw from the base
distribution, pushed through the flow, conditioned only on ``X`` and (for
zone-blocked updates) on the *complement* of the block being proposed. A
block's own current value is never fed back into its own proposal -- doing
so would make the "proposal" secretly depend on chain state, which breaks
the independence-sampler structure the acceptance ratio assumes. Both the
forward sample density and the reverse (current-point) density are computed
by inverting the same deterministic flow; there is no ODE, no trace
estimator, and no other source of randomness in a density evaluation.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .graph import Graph
from .flow import GraphCouplingLayer, masked_normal_logpdf, masked_student_t_logpdf


def _edge_softmax(logits: jax.Array, receivers: jax.Array, n_nodes: int) -> jax.Array:
    """Normalize edge logits independently over incoming edges per node."""
    maxima = jnp.full((n_nodes,), -jnp.inf).at[receivers].max(logits)
    exponentials = jnp.exp(logits - maxima[receivers])
    normalizers = jnp.zeros((n_nodes,)).at[receivers].add(exponentials)
    return exponentials / normalizers[receivers]


class GATLayer(eqx.Module):
    """Single graph-attention message-passing layer.

    Attention can down-weight edges that cross sharp feature discontinuities
    (e.g. morphological boundaries for the spot graph), which is why this
    conditioner is a GAT rather than a GCN or a grid-binned CNN: a uniform or
    grid-based aggregator would blur exactly the edges the loading MRF prior
    treats as breakable.
    """

    projection: eqx.nn.Linear
    attention: eqx.nn.Linear
    negative_slope: float = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, key: jax.Array):
        """Initialize projection and pairwise attention weights."""
        projection_key, attention_key = jax.random.split(key)
        self.projection = eqx.nn.Linear(in_size, out_size, key=projection_key)
        self.attention = eqx.nn.Linear(2 * out_size, 1, key=attention_key)
        self.negative_slope = 0.2

    def edge_attention(self, features: jax.Array, graph: Graph) -> jax.Array:
        """Return this layer's normalized per-edge attention weights.

        Exposed separately from :meth:`__call__` so callers can read off the
        attention structure itself -- e.g. to derive morphological zone
        partitions from the same edges the network already treats as weak
        (see :mod:`amortized_mcmc.zoning`) -- without recomputing it by hand.
        """
        projected = jax.vmap(self.projection)(features)
        pair_features = jnp.concatenate(
            (projected[graph.senders], projected[graph.receivers]), axis=-1
        )
        logits = jax.vmap(self.attention)(pair_features).squeeze(-1)
        logits = jax.nn.leaky_relu(logits, negative_slope=self.negative_slope)
        return _edge_softmax(logits, graph.receivers, features.shape[0])

    def __call__(self, features: jax.Array, graph: Graph) -> jax.Array:
        """Aggregate sender features into receiver nodes with attention."""
        projected = jax.vmap(self.projection)(features)
        weights = self.edge_attention(features, graph)
        aggregated = jnp.zeros_like(projected).at[graph.receivers].add(
            weights[:, None] * projected[graph.senders]
        )
        return jax.nn.elu(aggregated)


def _checkerboard_masks(always_passive: jax.Array, depth: int) -> tuple[jax.Array, ...]:
    """Build the per-layer passive-node masks for a graph coupling stack.

    Nodes marked ``always_passive`` (context that is never sampled, e.g. the
    complement of a zone block) stay passive in every layer. The remaining
    "free" nodes are split into two fixed groups by index parity; each layer
    transforms one group while treating the other -- plus every always-passive
    node -- as passive context. Alternating which half is active across at
    least two layers lets cross-node dependence within the free set flow
    through the conditioner while every individual layer's log-det stays a
    diagonal sum over only that layer's active nodes.
    """
    n_nodes = always_passive.shape[0]
    free = ~always_passive
    parity = (jnp.arange(n_nodes) % 2 == 0) & free
    other = free & ~parity
    return tuple(~(parity if index % 2 == 0 else other) for index in range(depth))


class _GraphCouplingHead(eqx.Module):
    """Shared machinery for a GAT-conditioned coupling flow over node arrays.

    Subclasses fix how the per-node context embedding is built (spot
    coordinates + expression for Head 1, pathway summary statistics for Head
    2) but share sampling, inversion, and masked base-density logic.
    """

    layers: tuple[GraphCouplingLayer, ...]
    node_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    base: str = eqx.field(static=True)
    degrees_of_freedom: float = eqx.field(static=True)

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        depth: int,
        gat_depth: int,
        key: jax.Array,
        base: str,
        degrees_of_freedom: float,
    ):
        """Build ``depth`` coupling layers, each with its own small GAT stack.

        ``hidden_dim`` is both the GAT hidden width and the per-node context
        embedding width the subclass's own ``embed`` layer must produce.
        """
        if depth < 2:
            raise ValueError("depth must be at least 2 so every free node is transformed at least once")
        if gat_depth < 1:
            raise ValueError("gat_depth must be positive")
        if base not in {"normal", "student_t"}:
            raise ValueError("base must be 'normal' or 'student_t'")
        layer_keys = jax.random.split(key, depth)
        layers = []
        node_input_dim = node_dim + hidden_dim + 1  # masked value, context embedding, passive flag
        for layer_key in layer_keys:
            gat_keys = jax.random.split(layer_key, gat_depth + 1)
            message_passing = []
            in_size = node_input_dim
            for gat_key in gat_keys[:-1]:
                message_passing.append(GATLayer(in_size, hidden_dim, gat_key))
                in_size = hidden_dim
            layers.append(GraphCouplingLayer(message_passing, hidden_dim, node_dim, gat_keys[-1]))
        self.layers = tuple(layers)
        self.node_dim = node_dim
        self.depth = depth
        self.base = base
        self.degrees_of_freedom = degrees_of_freedom

    def _base_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """Draw base noise from the configured heavy- or light-tailed base."""
        if self.base == "normal":
            return jax.random.normal(key, shape)
        return jax.random.t(key, self.degrees_of_freedom, shape)

    def _base_logpdf(self, value: jax.Array, row_mask: jax.Array) -> jax.Array:
        """Evaluate the masked base density over free (sampled) rows only."""
        if self.base == "normal":
            return masked_normal_logpdf(value, row_mask)
        return masked_student_t_logpdf(value, row_mask, self.degrees_of_freedom)

    def _forward(self, base_value: jax.Array, always_passive: jax.Array, complement: jax.Array, node_context: jax.Array, graph: Graph) -> tuple[jax.Array, jax.Array]:
        """Push base noise to data space, holding always-passive rows fixed."""
        value = jnp.where(always_passive[:, None], complement, base_value)
        masks = _checkerboard_masks(always_passive, self.depth)
        log_det = jnp.asarray(0.0, dtype=value.dtype)
        for layer, passive in zip(self.layers, masks):
            value, layer_log_det = layer.forward(value, passive, node_context, graph)
            log_det = log_det + layer_log_det
        return value, log_det

    def _inverse(self, value: jax.Array, always_passive: jax.Array, node_context: jax.Array, graph: Graph) -> tuple[jax.Array, jax.Array]:
        """Invert a data-space value back to base space."""
        masks = _checkerboard_masks(always_passive, self.depth)
        base_value = value
        log_det = jnp.asarray(0.0, dtype=value.dtype)
        for layer, passive in zip(self.layers[::-1], masks[::-1]):
            base_value, layer_log_det = layer.inverse(base_value, passive, node_context, graph)
            log_det = log_det + layer_log_det
        return base_value, log_det

    def logdensity(self, value: jax.Array, always_passive: jax.Array, node_context: jax.Array, graph: Graph) -> jax.Array:
        """Evaluate ``log q(value_free | X, value_complement)`` exactly."""
        base_value, inverse_log_det = self._inverse(value, always_passive, node_context, graph)
        return self._base_logpdf(base_value, ~always_passive) + inverse_log_det

    def sample_and_logdensity(
        self, key: jax.Array, always_passive: jax.Array, complement: jax.Array, node_context: jax.Array, graph: Graph
    ) -> tuple[jax.Array, jax.Array]:
        """Draw a fresh proposal for the free nodes and score it exactly."""
        base_value = self._base_sample(key, complement.shape)
        value, log_det = self._forward(base_value, always_passive, complement, node_context, graph)
        base_logpdf = self._base_logpdf(base_value, ~always_passive)
        return value, base_logpdf - log_det

    def edge_attention(self, node_context: jax.Array, graph: Graph) -> jax.Array:
        """Average this head's first coupling layer's GAT attention over edges.

        Used to derive zone-blocked update partitions that follow the
        network's own learned affinity/attention structure (per the model
        spec's "morphological zones read off the affinity/attention
        structure") instead of an arbitrary ordering -- see
        :mod:`amortized_mcmc.zoning`. All nodes are treated as passive here
        (their own value channel is a zero placeholder throughout, since no
        chain state need exist yet to read off zones from), so the returned
        weights reflect only ``node_context`` (the fixed embedding of ``X``),
        exactly the tissue/expression structure zoning should follow. A
        freshly initialized (untrained) head simply reflects its random
        initialization instead of anything learned.
        """
        n_nodes = node_context.shape[0]
        zeros = jnp.zeros((n_nodes, self.node_dim), dtype=node_context.dtype)
        always_passive = jnp.ones((n_nodes,), dtype=bool)
        features = jnp.concatenate((zeros, node_context, always_passive[:, None].astype(node_context.dtype)), axis=-1)
        hops = []
        for message_layer in self.layers[0].message_passing:
            hops.append(message_layer.edge_attention(features, graph))
            features = message_layer(features, graph)
        return jnp.mean(jnp.stack(hops, axis=0), axis=0)


class SpatialProposal(_GraphCouplingHead):
    """Head 1: GAT-conditioned coupling flow over the spot graph for ``L``.

    Conditioning context per spot is a fixed embedding of ``X`` and spatial
    coordinates; cross-spot dependence and complement conditioning both flow
    through GAT message passing inside the coupling layers, never through a
    dependence on the block's own current value.
    """

    embed: eqx.nn.Linear
    coordinate_dim: int = eqx.field(static=True)

    def __init__(
        self,
        H: int,
        observation_dim: int,
        hidden_dim: int = 32,
        depth: int = 4,
        *,
        key: jax.Array,
        coordinate_dim: int = 2,
        gat_depth: int = 2,
        base: str = "normal",
        degrees_of_freedom: float = 4.0,
    ):
        """Initialize the spot-context embedding and coupling-layer stack."""
        embed_key, head_key = jax.random.split(key)
        super().__init__(H, hidden_dim, depth, gat_depth, head_key, base, degrees_of_freedom)
        self.embed = eqx.nn.Linear(observation_dim + coordinate_dim, hidden_dim, key=embed_key)
        self.coordinate_dim = coordinate_dim

    @property
    def H(self) -> int:
        """Return the configured CSP truncation width."""
        return self.node_dim

    def node_context(self, X: jax.Array, coordinates: jax.Array | None = None) -> jax.Array:
        """Embed normalized per-spot counts and coordinates into context."""
        if coordinates is None:
            coordinates = jnp.zeros((X.shape[0], self.coordinate_dim), dtype=X.dtype)
        normalized_X = jnp.log1p(X)
        normalized_X = normalized_X / (jnp.mean(normalized_X, axis=0, keepdims=True) + 1e-6)
        raw = jnp.concatenate((normalized_X, coordinates), axis=-1)
        return jax.vmap(self.embed)(raw)

    def log_q(self, value: jax.Array, X: jax.Array, always_passive: jax.Array, graph: Graph, coordinates: jax.Array | None = None) -> jax.Array:
        """Evaluate the exact proposal density at a candidate ``L``."""
        return self.logdensity(value, always_passive, self.node_context(X, coordinates), graph)

    def attention_edge_weights(self, X: jax.Array, graph: Graph, coordinates: jax.Array | None = None) -> jax.Array:
        """Read off this head's spot-graph attention weights for zoning."""
        return self.edge_attention(self.node_context(X, coordinates), graph)

    def sample_and_log_q(
        self,
        key: jax.Array,
        X: jax.Array,
        always_passive: jax.Array,
        complement: jax.Array,
        graph: Graph,
        coordinates: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample a fresh ``L`` for the free spots and return its exact density.

        ``complement`` must carry the current values at ``always_passive``
        spots; values at free spots are ignored (they are drawn fresh).
        """
        return self.sample_and_logdensity(key, always_passive, complement, self.node_context(X, coordinates), graph)


class LoadingProposal(_GraphCouplingHead):
    """Head 2: GAT-conditioned coupling flow over the pathway graph for ``F``.

    This is the scientifically primary head. Its base distribution is
    Student-t by default (not Gaussian): the Horseshoe prior gives ``F``
    heavy-tailed conditionals, and a Lipschitz coupling transform cannot
    manufacture heavy tails from a light base, so the proposal's tails must
    already dominate the target's for the independence sampler to stay
    geometrically ergodic.
    """

    element_encoder: eqx.nn.MLP
    embed: eqx.nn.Linear
    pooled_dim: int = eqx.field(static=True)

    def __init__(
        self,
        H: int,
        n_genes: int,
        hidden_dim: int = 64,
        depth: int = 4,
        *,
        key: jax.Array,
        gat_depth: int = 2,
        base: str = "student_t",
        degrees_of_freedom: float = 4.0,
        pooled_dim: int = 16,
    ):
        """Initialize the gene-context encoder and coupling-layer stack.

        Head 2 is the scientifically primary head, so its conditioning on
        ``X`` is deliberately not a couple of hand-picked summary statistics
        (mean/variance): a shared per-spot encoder (a DeepSets-style
        permutation-invariant pooling over spots) maps every spot's count
        for a gene to a learned vector, mean- and std-pooled across spots
        into a per-gene embedding. This lets training shape the featurization
        of the full expression column, on par with Head 1's use of a spot's
        full observed count vector, rather than compressing it by hand.
        """
        del n_genes  # kept for interface symmetry with SpatialProposal; genes are not fixed-width here
        element_key, embed_key, head_key = jax.random.split(key, 3)
        super().__init__(H, hidden_dim, depth, gat_depth, head_key, base, degrees_of_freedom)
        self.element_encoder = eqx.nn.MLP(1, pooled_dim, hidden_dim, 2, key=element_key)
        self.embed = eqx.nn.Linear(2 * pooled_dim, hidden_dim, key=embed_key)
        self.pooled_dim = pooled_dim

    @property
    def H(self) -> int:
        """Return the configured CSP truncation width."""
        return self.node_dim

    def node_context(self, X: jax.Array) -> jax.Array:
        """Pool a learned per-spot encoding of counts into a per-gene embedding."""
        n_spots, n_genes = X.shape
        log_counts = jnp.log1p(X).reshape(-1, 1)
        encoded = jax.vmap(self.element_encoder)(log_counts).reshape(n_spots, n_genes, self.pooled_dim)
        mean_pool = jnp.mean(encoded, axis=0)
        # jnp.std has an undefined (NaN) gradient at exactly zero variance,
        # which a constant-across-spots gene column (e.g. an all-zero count
        # column, common in sparse data) hits often enough in practice to
        # poison training; sqrt(var + eps) keeps the gradient finite there.
        std_pool = jnp.sqrt(jnp.var(encoded, axis=0) + 1e-6)
        summary = jnp.concatenate((mean_pool, std_pool), axis=-1)
        return jax.vmap(self.embed)(summary)

    def log_q(self, value: jax.Array, X: jax.Array, always_passive: jax.Array, pathway_graph: Graph) -> jax.Array:
        """Evaluate the exact proposal density at a candidate ``F``."""
        return self.logdensity(value, always_passive, self.node_context(X), pathway_graph)

    def attention_edge_weights(self, X: jax.Array, pathway_graph: Graph) -> jax.Array:
        """Read off this head's pathway-graph attention weights for zoning."""
        return self.edge_attention(self.node_context(X), pathway_graph)

    def sample_and_log_q(
        self, key: jax.Array, X: jax.Array, always_passive: jax.Array, complement: jax.Array, pathway_graph: Graph
    ) -> tuple[jax.Array, jax.Array]:
        """Sample a fresh ``F`` for the free genes and return its exact density.

        ``complement`` must carry the current values at ``always_passive``
        genes; values at free genes are ignored (they are drawn fresh).
        """
        return self.sample_and_logdensity(key, always_passive, complement, self.node_context(X), pathway_graph)


class AmortizedProposal(NamedTuple):
    """Pair of independent spatial and loading proposal heads."""
    spatial: SpatialProposal
    loading: LoadingProposal
