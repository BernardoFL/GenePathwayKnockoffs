"""Spatial graph construction for irregular spot layouts.

Graphs are represented as directed edge lists with both directions of every
undirected neighbor relation. The ``Graph`` pytree can therefore travel
through JAX transformations alongside model states.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import numpy as np
from scipy.spatial import Delaunay


@register_pytree_node_class
@dataclass(frozen=True)
class Graph:
    """Directed spot graph with symmetric edge storage.

    Attributes:
        senders: Integer source-node indices with shape ``(E,)``.
        receivers: Integer destination-node indices with shape ``(E,)``.
        distances: Euclidean edge lengths with shape ``(E,)``.
    """

    senders: jnp.ndarray
    receivers: jnp.ndarray
    distances: jnp.ndarray

    @property
    def n_edges(self) -> int:
        """Return the number of directed edges in the graph."""
        return int(self.senders.shape[0])

    def tree_flatten(self):
        """Expose array fields to JAX pytree transformations."""
        return (self.senders, self.receivers, self.distances), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct a graph from flattened pytree children."""
        return cls(*children)


def _directed_edges(edges: np.ndarray, coordinates: np.ndarray) -> Graph:
    """Canonicalize undirected edges and return both directed orientations."""
    edges = np.asarray(edges, dtype=np.int32)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    reverse = edges[:, ::-1]
    directed = np.concatenate((edges, reverse), axis=0)
    deltas = coordinates[directed[:, 0]] - coordinates[directed[:, 1]]
    distances = np.linalg.norm(deltas, axis=1).astype(np.float32)
    return Graph(jnp.asarray(directed[:, 0]), jnp.asarray(directed[:, 1]), jnp.asarray(distances))


def build_graph(coordinates: np.ndarray, method: str = "knn", k: int = 6) -> Graph:
    """Build a k-NN or Delaunay graph from spot coordinates.

    Args:
        coordinates: Numeric array with shape ``(S, D)``.
        method: ``"knn"`` for nearest neighbors or ``"delaunay"`` for a 2-D
            triangulation.
        k: Number of nearest neighbors for the k-NN method.

    Returns:
        A JAX-compatible :class:`Graph` containing directed edges.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("coordinates must have shape (n_spots, n_dims), with at least two spots")
    if method == "knn":
        n_spots = coordinates.shape[0]
        if not 1 <= k < n_spots:
            raise ValueError("k must satisfy 1 <= k < n_spots")
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
        neighbors = np.argsort(distances, axis=1)[:, 1 : k + 1]
        edges = np.stack(
            (np.repeat(np.arange(n_spots), k), neighbors.reshape(-1)), axis=1
        )
    elif method == "delaunay":
        if coordinates.shape[1] != 2:
            raise ValueError("Delaunay triangulation currently requires 2D coordinates")
        simplices = Delaunay(coordinates).simplices
        edges = np.concatenate(
            [simplices[:, [0, 1]], simplices[:, [0, 2]], simplices[:, [1, 2]]], axis=0
        )
    else:
        raise ValueError("method must be 'knn' or 'delaunay'")
    return _directed_edges(edges, coordinates)


def build_pathway_graph(affinity: np.ndarray, threshold: float = 0.0) -> Graph:
    """Build a directed edge-list ``Graph`` from a weighted gene affinity matrix.

    Edge weights (rather than Euclidean distances) are stored in the
    ``distances`` field. This graph is the pathway-annotation graph
    ``A_path`` that conditions Head 2's coupling flow and the pathway prior;
    see :func:`assert_graphs_not_aliased` for why it must stay a distinct
    object from any data-driven co-expression graph used by the knockoff
    filter.
    """
    affinity = np.asarray(affinity, dtype=np.float32)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be a square (n_genes, n_genes) matrix")
    senders, receivers = np.nonzero(np.abs(affinity) > threshold)
    keep = senders != receivers
    senders, receivers = senders[keep], receivers[keep]
    weights = affinity[senders, receivers]
    return Graph(jnp.asarray(senders, dtype=jnp.int32), jnp.asarray(receivers, dtype=jnp.int32), jnp.asarray(weights))


def assert_graphs_not_aliased(pathway_graph: Graph, knockoff_graph: Graph) -> None:
    """Assert the pathway-annotation graph and knockoff test graph are distinct.

    The pathway prior smooths ``F`` through the *annotation* graph
    ``A_path``; the knockoff filter tests edge-level statistics defined by a
    separate, *data-driven* co-expression graph ``A_jk``. If the same graph
    object or edge set backed both, the prior would place mass on exactly
    the structure the filter is meant to discover, biasing discovery toward
    annotated pathways and undercutting the finite-sample FDR guarantee.
    Call this wherever both graphs are assembled for a run.
    """
    if pathway_graph is knockoff_graph:
        raise ValueError("pathway graph and knockoff test graph must not be the same object")
    same_shape = pathway_graph.senders.shape == knockoff_graph.senders.shape
    if same_shape and bool(jnp.all(pathway_graph.senders == knockoff_graph.senders)) and bool(
        jnp.all(pathway_graph.receivers == knockoff_graph.receivers)
    ):
        raise ValueError("pathway graph and knockoff test graph must not share the same edge set")
