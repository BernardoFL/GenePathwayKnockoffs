#!/usr/bin/env python
"""Train both amortizer heads by maximum-likelihood flow density estimation.

Per spec §4: each training example is one prior-simulation draw ``(L, F, X)``
from :mod:`scripts.simulate_prior` -- *not* a Markov transition pair, and
there is no conditional flow matching and no reference MCMC anywhere in this
loop. The objective is the forward-KL / maximum-likelihood flow loss

    E[-log q_phi(L | X, L_complement)] + E[-log q_phi(F | X, F_complement)]

whose optimum is the true posterior full conditional at every context. Each
step draws a random context mask per example: with probability
``--empty-context-prob`` the mask is empty (training the global-jump mode
``q_phi(L | X)``); otherwise a random fraction of nodes is held out as fixed
context (training the zone-blocked conditional mode ``q_phi(L_B | X,
L_-B)``). One network serves both moves.
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

from amortized_mcmc import LoadingProposal, SpatialProposal, build_pathway_graph
from amortized_mcmc.graph import Graph


def parse_args() -> argparse.Namespace:
    """Parse training and checkpoint settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="npz shard from scripts/simulate_prior.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loading-hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4, help="number of graph coupling layers per head")
    parser.add_argument("--gat-depth", type=int, default=2, help="GAT message-passing hops per coupling layer")
    parser.add_argument("--spatial-base", choices=("normal", "student_t"), default="normal")
    parser.add_argument("--loading-base", choices=("normal", "student_t"), default="student_t")
    parser.add_argument("--degrees-of-freedom", type=float, default=4.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--examples-per-step", type=int, default=8)
    parser.add_argument("--empty-context-prob", type=float, default=0.3, help="fraction of steps trained as a global (context-free) jump")
    parser.add_argument("--min-context-frac", type=float, default=0.1)
    parser.add_argument("--max-context-frac", type=float, default=0.9)
    return parser.parse_args()


def load_shard(path: Path):
    """Load a §6-schema npz shard and reconstruct per-example spot graphs."""
    with np.load(path, allow_pickle=True) as data:
        required = {"n_node", "n_edge", "coords", "edge_index", "X", "L", "F", "A_path", "gene_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"training npz is missing: {sorted(missing)}")
        coords = np.asarray(data["coords"], dtype=np.float32)
        X = np.asarray(data["X"], dtype=np.float32)
        L = np.asarray(data["L"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int32)
        n_node = np.asarray(data["n_node"], dtype=np.int32)
        n_edge = np.asarray(data["n_edge"], dtype=np.int32)
        F = np.asarray(data["F"], dtype=np.float32)
        A_path = np.asarray(data["A_path"], dtype=np.float32)
        gene_ids = np.asarray(data["gene_ids"])
    if edge_index.shape != (2, int(n_edge.sum())):
        raise ValueError("edge_index must have shape (2, sum(n_edge)) per the §6 schema")
    if F.shape[0] != n_node.shape[0]:
        raise ValueError("F's leading axis must equal the number of examples N")
    if F.shape[1] != A_path.shape[0] or F.shape[1] != gene_ids.shape[0]:
        raise ValueError("F, A_path, and gene_ids must share the same gene axis")

    graphs = []
    node_offset = edge_offset = 0
    for node_count, edge_count in zip(n_node, n_edge):
        node_slice = slice(node_offset, node_offset + int(node_count))
        edge_slice = slice(edge_offset, edge_offset + int(edge_count))
        local_senders = edge_index[0, edge_slice] - node_offset
        local_receivers = edge_index[1, edge_slice] - node_offset
        graphs.append(
            jraph.GraphsTuple(
                nodes={"coords": jnp.asarray(coords[node_slice]), "X": jnp.asarray(X[node_slice]), "L": jnp.asarray(L[node_slice])},
                edges=None,
                senders=jnp.asarray(local_senders),
                receivers=jnp.asarray(local_receivers),
                globals=None,
                n_node=jnp.asarray([node_count], dtype=jnp.int32),
                n_edge=jnp.asarray([edge_count], dtype=jnp.int32),
            )
        )
        node_offset += int(node_count)
        edge_offset += int(edge_count)
    # Round-trip through jraph so storage and loader agree on its canonical
    # flattened-graph convention rather than a hand-rolled segmentation.
    graphs = jraph.unbatch(jraph.batch(graphs))
    pathway_graph = build_pathway_graph(A_path)
    return graphs, jnp.asarray(F), pathway_graph, gene_ids


def graph_to_model_graph(graph: jraph.GraphsTuple) -> Graph:
    """Convert a jraph edge list to the package's distance-aware Graph type."""
    coords = np.asarray(graph.nodes["coords"])
    distances = np.linalg.norm(coords[np.asarray(graph.senders)] - coords[np.asarray(graph.receivers)], axis=-1)
    return Graph(graph.senders, graph.receivers, jnp.asarray(distances, dtype=jnp.float32))


def sample_context_mask(key: jax.Array, n_nodes: int, empty_prob: float, min_frac: float, max_frac: float) -> jax.Array:
    """Draw a random always-passive (context) node mask for one training step.

    With probability ``empty_prob`` the mask is empty, training the
    global-jump mode ``q_phi(. | X)``; otherwise a random fraction of nodes
    is held out as fixed context, training the zone-blocked conditional
    mode ``q_phi(free | X, context)``. An empty context is exactly the
    ``rho=1`` special case described in §4.
    """
    empty_key, frac_key, mask_key = jax.random.split(key, 3)
    is_empty = jax.random.uniform(empty_key) < empty_prob
    frac = jax.random.uniform(frac_key, minval=min_frac, maxval=max_frac)
    always_passive = jax.random.uniform(mask_key, (n_nodes,)) < frac
    return always_passive & (~is_empty)


def save_checkpoint(output, spatial, loading, spatial_state, loading_state, epoch, losses, metadata):
    """Save both Equinox heads, optimizer leaves, metrics, and metadata."""
    output.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(output / f"spatial_proposal_{epoch:07d}.eqx", spatial)
    eqx.tree_serialise_leaves(output / f"loading_proposal_{epoch:07d}.eqx", loading)
    np.savez_compressed(output / f"optimizer_{epoch:07d}.npz", *jax.tree_util.tree_leaves((spatial_state, loading_state)))
    (output / f"metrics_{epoch:07d}.json").write_text(json.dumps({**metadata, "epoch": epoch, "spatial_nll": losses[0], "loading_nll": losses[1]}, indent=2) + "\n")


def main():
    """Fit both heads' exact flow densities to prior-simulation draws."""
    args = parse_args()
    graphs, F_values, pathway_graph, gene_ids = load_shard(args.data)
    n_genes = F_values.shape[1]
    H = F_values.shape[2]
    observation_dim = graphs[0].nodes["X"].shape[-1]
    key = jax.random.PRNGKey(args.seed)
    spatial_key, loading_key, key = jax.random.split(key, 3)
    spatial = SpatialProposal(
        H, observation_dim, args.hidden_dim, args.depth, key=spatial_key, gat_depth=args.gat_depth, base=args.spatial_base, degrees_of_freedom=args.degrees_of_freedom
    )
    loading = LoadingProposal(
        H, n_genes, args.loading_hidden_dim, args.depth, key=loading_key, gat_depth=args.gat_depth, base=args.loading_base, degrees_of_freedom=args.degrees_of_freedom
    )
    optimizer = optax.adam(args.learning_rate)
    spatial_state = optimizer.init(eqx.filter(spatial, eqx.is_array))
    loading_state = optimizer.init(eqx.filter(loading, eqx.is_array))
    history = []

    for epoch in range(1, args.epochs + 1):
        key, step_key = jax.random.split(key)
        permutation = np.asarray(jax.random.permutation(step_key, len(graphs))[: args.examples_per_step])
        mask_keys = jax.random.split(step_key, 2 * len(permutation))
        selected = [(graphs[int(index)], F_values[int(index)]) for index in permutation]

        def spatial_loss_fn(model):
            """Average negative log-density of the true L under random context masks."""
            losses = []
            for offset, (graph, _) in enumerate(selected):
                nodes = graph.nodes
                model_graph = graph_to_model_graph(graph)
                always_passive = sample_context_mask(mask_keys[offset], nodes["L"].shape[0], args.empty_context_prob, args.min_context_frac, args.max_context_frac)
                losses.append(-model.log_q(nodes["L"], nodes["X"], always_passive, model_graph, nodes["coords"]))
            return jnp.mean(jnp.asarray(losses))

        def loading_loss_fn(model):
            """Average negative log-density of the true F under random context masks."""
            losses = []
            for offset, (graph, F_true) in enumerate(selected):
                nodes = graph.nodes
                always_passive = sample_context_mask(mask_keys[len(selected) + offset], n_genes, args.empty_context_prob, args.min_context_frac, args.max_context_frac)
                losses.append(-model.log_q(F_true, nodes["X"], always_passive, pathway_graph))
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
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch}: spatial NLL {history[-1][0]:.4f}, loading NLL {history[-1][1]:.4f}", flush=True)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                args.output, spatial, loading, spatial_state, loading_state, epoch, history[-1],
                {"H": H, "n_genes": n_genes, "n_graphs": len(graphs), "variable_spot_count": True},
            )
            np.save(args.output / "loss_history.npy", np.asarray(history, dtype=np.float32))


if __name__ == "__main__":
    main()
