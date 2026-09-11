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
