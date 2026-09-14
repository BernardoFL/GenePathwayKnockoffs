"""Morphological/attention-derived zone partitioning for zone-blocked moves.

The model spec calls for zone-blocked updates over "morphological zones
read off the affinity/attention structure" -- not an arbitrary ordering.
This module reads a (optionally attention-weighted) graph's own
connectivity via a lightweight recursive spectral (Fiedler-vector)
bisection, so zone boundaries fall on weakly-connected/weakly-attended
edges -- the same edges a trained head already treats as weak -- rather
than cutting through strongly coupled neighborhoods.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _weighted_laplacian(n_nodes: int, senders: np.ndarray, receivers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Build a dense symmetric weighted graph Laplacian from a directed edge list."""
    affinity = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    affinity[senders, receivers] = np.maximum(affinity[senders, receivers], weights)
    affinity[receivers, senders] = np.maximum(affinity[receivers, senders], weights)
    degree = np.diag(affinity.sum(axis=1))
    return degree - affinity


def _fiedler_bisection(laplacian: np.ndarray, node_indices: np.ndarray) -> list[np.ndarray]:
    """Split ``node_indices`` in two by the sign of the local Fiedler vector."""
    if len(node_indices) < 2:
        return [node_indices]
    sub_laplacian = laplacian[np.ix_(node_indices, node_indices)]
    eigenvalues, eigenvectors = np.linalg.eigh(sub_laplacian)
    fiedler = eigenvectors[:, 1] if sub_laplacian.shape[0] > 1 else eigenvectors[:, 0]
    threshold = np.median(fiedler)
    group_a = node_indices[fiedler >= threshold]
    group_b = node_indices[fiedler < threshold]
    if len(group_a) == 0 or len(group_b) == 0:
        # A degenerate split (e.g. disconnected components with identical
        # Fiedler values) falls back to a positional split so no zone is
        # ever returned empty.
        midpoint = len(node_indices) // 2
        group_a, group_b = node_indices[:midpoint], node_indices[midpoint:]
    return [group_a, group_b]


def spectral_zone_masks(
    n_nodes: int,
    senders,
    receivers,
    n_zones: int,
    weights=None,
) -> jnp.ndarray:
    """Partition nodes into ``n_zones`` groups by recursive Fiedler bisection.

    ``weights`` should be per-edge attention scores from a trained head
    (:meth:`amortized_mcmc.models.SpatialProposal.attention_edge_weights` or
    the ``LoadingProposal`` analogue) when available; omitting it falls back
    to an unweighted graph, which still respects real connectivity (unlike
    an arbitrary coordinate ordering) but not learned affinity.

    Returns a boolean array of shape ``(n_zones, n_nodes)`` where row ``r``
    is ``True`` everywhere *outside* zone ``r``, matching the
    ``mixture_step``/``propose_*_block`` complement/context convention.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    n_zones = max(1, min(n_zones, n_nodes))
    senders = np.asarray(senders)
    receivers = np.asarray(receivers)
    if weights is None:
        weights = np.ones(senders.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    laplacian = _weighted_laplacian(n_nodes, senders, receivers, weights)

    groups = [np.arange(n_nodes)]
    while len(groups) < n_zones:
        # Always split the currently-largest group so zones stay balanced.
        largest_index = int(np.argmax([len(group) for group in groups]))
        largest = groups.pop(largest_index)
        groups.extend(_fiedler_bisection(laplacian, largest))
    groups = groups[:n_zones]

    masks = np.ones((len(groups), n_nodes), dtype=bool)
    for row, group in enumerate(groups):
        masks[row, group] = False
    return jnp.asarray(masks)
