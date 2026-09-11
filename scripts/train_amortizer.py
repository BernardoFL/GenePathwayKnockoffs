#!/usr/bin/env python
"""Train both amortizer heads on variable-size joint-reference graphs.

The input NPZ uses flattened jraph storage: ``coords``, ``X``, ``L_current``
and ``L_prime`` are concatenated over spots; ``edge_index`` is globally offset;
``n_node`` and ``n_edge`` delimit examples. ``F_current`` and ``F_prime``
remain ``(N, G, H)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax

from amortized_mcmc import LoadingProposal, SpatialProposal, build_graph, conditional_flow_matching_loss
from amortized_mcmc.graph import Graph


def parse_args():
    """Parse variable-graph training and checkpoint settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loading-hidden-dim", type=int, default=128)
    parser.add_argument("--gat-depth", type=int, default=2)
    parser.add_argument("--loading-depth", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--graphs-per-step", type=int, default=8)
    return parser.parse_args()


def load_graph_batch(path: Path, H: int):
    """Load flattened NPZ storage and reconstruct graphs with jraph.

    The returned graphs are unbatched ``GraphsTuple`` objects so each example
    can retain its own number of spots. Batching remains available through
    ``jraph.batch`` for callers that want a combined message-passing graph.
    """
    with np.load(path) as data:
        required = {"coords", "X", "L_current", "L_prime", "edge_index", "n_node", "n_edge", "F_current", "F_prime"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"training NPZ is missing: {sorted(missing)}")
        coords = np.asarray(data["coords"], dtype=np.float32)
        X = np.asarray(data["X"], dtype=np.float32)
        L_current = np.asarray(data["L_current"], dtype=np.float32)
        L_prime = np.asarray(data["L_prime"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int32)
        n_node = np.asarray(data["n_node"], dtype=np.int32)
        n_edge = np.asarray(data["n_edge"], dtype=np.int32)
        F_current = np.asarray(data["F_current"], dtype=np.float32)
        F_prime = np.asarray(data["F_prime"], dtype=np.float32)
    if L_current.shape != L_prime.shape or L_current.shape[0] != coords.shape[0]:
        raise ValueError("flattened node arrays must have matching first dimensions")
    if L_current.shape[-1] != H or F_current.shape[-1] != H or F_current.shape != F_prime.shape:
        raise ValueError("L and F transition widths/shapes must match H")
    if F_current.shape[0] != n_node.shape[0] or len(n_node) != len(n_edge):
        raise ValueError("F arrays and graph boundary metadata must share N")
    if edge_index.shape != (int(n_edge.sum()), 2):
        raise ValueError("edge_index must have shape (sum(n_edge), 2)")

    graphs = []
    node_offset = edge_offset = 0
    for index, (node_count, edge_count) in enumerate(zip(n_node, n_edge)):
        node_slice = slice(node_offset, node_offset + int(node_count))
        edge_slice = slice(edge_offset, edge_offset + int(edge_count))
        local_edges = edge_index[edge_slice] - node_offset
        graphs.append(
            jraph.GraphsTuple(
                nodes={
                    "coords": jnp.asarray(coords[node_slice]),
                    "X": jnp.asarray(X[node_slice]),
                    "L_current": jnp.asarray(L_current[node_slice]),
                    "L_prime": jnp.asarray(L_prime[node_slice]),
                },
                edges=None,
                senders=jnp.asarray(local_edges[:, 0]),
                receivers=jnp.asarray(local_edges[:, 1]),
                globals=None,
                n_node=jnp.asarray([node_count], dtype=jnp.int32),
                n_edge=jnp.asarray([edge_count], dtype=jnp.int32),
            )
        )
        node_offset += int(node_count)
        edge_offset += int(edge_count)
    # Batch and unbatch through jraph so the storage and loader follow its
    # canonical flattened-graph convention rather than custom segmentation.
    return jraph.unbatch(jraph.batch(graphs)), jnp.asarray(F_current), jnp.asarray(F_prime)


def graph_to_model_graph(graph: jraph.GraphsTuple) -> Graph:
    """Convert a jraph edge list to the package's distance-aware Graph type."""
    coords = np.asarray(graph.nodes["coords"])
    distances = np.linalg.norm(coords[np.asarray(graph.senders)] - coords[np.asarray(graph.receivers)], axis=-1)
    return Graph(graph.senders, graph.receivers, jnp.asarray(distances, dtype=jnp.float32))


def save_checkpoint(output, spatial, loading, spatial_state, loading_state, epoch, losses, metadata):
    """Save both Equinox heads, optimizer leaves, metrics, and metadata."""
    output.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(output / f"spatial_proposal_{epoch:07d}.eqx", spatial)
    eqx.tree_serialise_leaves(output / f"loading_proposal_{epoch:07d}.eqx", loading)
    np.savez_compressed(output / f"optimizer_{epoch:07d}.npz", *jax.tree_util.tree_leaves((spatial_state, loading_state)))
    (output / f"metrics_{epoch:07d}.json").write_text(json.dumps({**metadata, "epoch": epoch, "spatial_loss": losses[0], "loading_loss": losses[1]}, indent=2) + "\n")


def main():
    """Train both flow heads over sampled variable-size graph examples."""
    args = parse_args()
    graphs, F_current, F_prime = load_graph_batch(args.data, args.H)
    first_nodes = graphs[0].nodes
    spatial_key, loading_key, key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    spatial = SpatialProposal(args.H, first_nodes["X"].shape[-1], args.hidden_dim, args.gat_depth, key=spatial_key)
    loading = LoadingProposal(first_nodes["X"].shape[-1], F_current.shape[1], args.H, args.loading_hidden_dim, args.loading_depth, key=loading_key)
    optimizer = optax.adam(args.learning_rate)
    spatial_state = optimizer.init(eqx.filter(spatial, eqx.is_array))
    loading_state = optimizer.init(eqx.filter(loading, eqx.is_array))
    history = []

    for epoch in range(1, args.epochs + 1):
        key, step_key = jax.random.split(key)
        permutation = np.asarray(jax.random.permutation(step_key, len(graphs))[: args.graphs_per_step])
        selected = [(graphs[int(index)], F_current[int(index)], F_prime[int(index)]) for index in permutation]
        sample_keys = jax.random.split(step_key, max(2 * len(selected), 1))

        def spatial_loss_fn(model):
            """Average CFM loss for the selected variable-size spatial graphs."""
            losses = []
            for index, (graph, _, _) in enumerate(selected):
                nodes = graph.nodes
                model_graph = graph_to_model_graph(graph)
                base = jax.random.normal(sample_keys[index], nodes["L_current"].shape)
                field = lambda t, value: model.flow_field(
                    t, value, nodes["X"], nodes["L_current"], model_graph, nodes["coords"]
                )
                losses.append(conditional_flow_matching_loss(field, sample_keys[index], base, nodes["L_prime"]))
            return jnp.mean(jnp.asarray(losses))

        def loading_loss_fn(model):
            """Average CFM loss for loading transitions in the selected batch."""
            losses = []
            for index, (graph, F_i, F_target) in enumerate(selected):
                nodes = graph.nodes
                base = jax.random.normal(sample_keys[len(selected) + index], F_i.shape)
                field = lambda t, value: model.flow_field(t, value, nodes["X"], F_i)
                losses.append(conditional_flow_matching_loss(field, sample_keys[len(selected) + index], base, F_target))
            return jnp.mean(jnp.asarray(losses))

        spatial_loss = spatial_loss_fn(spatial)
        loading_loss = loading_loss_fn(loading)
        grads_spatial = eqx.filter_grad(spatial_loss_fn)(spatial)
        grads_loading = eqx.filter_grad(loading_loss_fn)(loading)
        updates, spatial_state = optimizer.update(grads_spatial, spatial_state, spatial)
        spatial = eqx.apply_updates(spatial, updates)
        updates, loading_state = optimizer.update(grads_loading, loading_state, loading)
        loading = eqx.apply_updates(loading, updates)
        history.append((float(spatial_loss), float(loading_loss)))
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(args.output, spatial, loading, spatial_state, loading_state, epoch, history[-1], {"H": args.H, "n_graphs": len(graphs), "variable_spot_count": True})
            np.save(args.output / "loss_history.npy", np.asarray(history, dtype=np.float32))


if __name__ == "__main__":
    main()
