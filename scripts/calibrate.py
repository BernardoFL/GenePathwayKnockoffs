#!/usr/bin/env python
"""Calibration harness for the trained amortized heads (spec §4 / §10.7).

Loads trained ``SpatialProposal``/``LoadingProposal`` checkpoints and a
prior-simulation npz shard (from ``scripts/simulate_prior.py`` -- ideally
one held out from training) and reports, for both heads and for both the
empty-context (global-jump) and a random partial-context (zone-blocked)
mode:

* **Empirical coverage** of nominal credible intervals built from q_phi's
  own Monte Carlo samples (mean/std over ``--n-samples`` draws), checked
  against the shard's true prior draws -- the deliverable §4 asks for
  instead of held-out NLL, which can look fine while coverage is bad.
* A **simulation-based-calibration (SBC) rank** diagnostic: the rank of the
  true point's own flow log-density among the same samples' log-densities
  under the same q_phi(. | X, context). A well-calibrated q places the true
  point among typical samples (ranks roughly uniform in [0, 1]); a q whose
  variance is too small systematically ranks the true point in the tail --
  exactly the failure mode the spec's "variance must dominate the true
  conditional's" requirement guards against, since a too-narrow q breaks
  the independence sampler's geometric ergodicity.
* For Head 1, both diagnostics **stratified by proximity to a morphological
  boundary** (a spot's total Laplace-MRF potential to its neighbors, from
  the shard's true ``L``) -- where the boundary-preservation claim is
  actually tested.

Coverage/rank numbers from a briefly- or lightly-trained checkpoint will
look poor; that reflects training compute, not a harness defect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from amortized_mcmc import LoadingProposal, SpatialProposal, build_pathway_graph, empirical_coverage
from amortized_mcmc.graph import Graph


def parse_args() -> argparse.Namespace:
    """Parse calibration-run settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="held-out npz shard from scripts/simulate_prior.py")
    parser.add_argument("--output", type=Path, default=None, help="optional path to write the JSON report")
    parser.add_argument("--spatial-checkpoint", type=Path, default=None)
    parser.add_argument("--loading-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=50, help="number of shard examples to evaluate (capped at the shard size)")
    parser.add_argument("--n-samples", type=int, default=64, help="Monte Carlo draws from q_phi per example")
    parser.add_argument("--context-frac", type=float, default=0.5, help="held-out node fraction for the zone-blocked evaluation mode")
    parser.add_argument("--coverage-level", type=float, default=0.9)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loading-hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--gat-depth", type=int, default=2)
    parser.add_argument("--spatial-base", choices=("normal", "student_t"), default="normal")
    parser.add_argument("--loading-base", choices=("normal", "student_t"), default="student_t")
    parser.add_argument("--degrees-of-freedom", type=float, default=4.0)
    return parser.parse_args()


def load_shard(path: Path):
    """Load a §6-schema npz shard and reconstruct per-example spot graphs."""
    with np.load(path, allow_pickle=True) as data:
        coords = np.asarray(data["coords"], dtype=np.float32)
        X = np.asarray(data["X"], dtype=np.float32)
        L = np.asarray(data["L"], dtype=np.float32)
        edge_index = np.asarray(data["edge_index"], dtype=np.int32)
        n_node = np.asarray(data["n_node"], dtype=np.int32)
        n_edge = np.asarray(data["n_edge"], dtype=np.int32)
        F = np.asarray(data["F"], dtype=np.float32)
        A_path = np.asarray(data["A_path"], dtype=np.float32)

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
    graphs = jraph.unbatch(jraph.batch(graphs))
    pathway_graph = build_pathway_graph(A_path)
    return graphs, jnp.asarray(F), pathway_graph


def graph_to_model_graph(graph: jraph.GraphsTuple) -> Graph:
    """Convert a jraph edge list to the package's distance-aware Graph type."""
    coords = np.asarray(graph.nodes["coords"])
    distances = np.linalg.norm(coords[np.asarray(graph.senders)] - coords[np.asarray(graph.receivers)], axis=-1)
    return Graph(graph.senders, graph.receivers, jnp.asarray(distances, dtype=jnp.float32))


def boundary_score(value: jax.Array, graph: Graph) -> jax.Array:
    """Sum of a node's incoming Laplace-MRF potential -- a boundary proxy.

    Nodes near a real morphological discontinuity have large true
    neighbor-to-neighbor differences; this reuses that same L1 potential
    (see ``amortized_mcmc.target.laplace_mrf_logprob``) as a per-node score
    to stratify calibration by boundary proximity.
    """
    edge_scores = jnp.sum(jnp.abs(value[graph.senders] - value[graph.receivers]), axis=-1)
    return jnp.zeros((value.shape[0],)).at[graph.receivers].add(edge_scores)


def evaluate_head(key, model, X, true_value, always_passive, context, graph, n_samples: int, *extra_args):
    """Draw ``n_samples`` from ``model`` and return truth/mean/std/rank arrays for free nodes.

    ``extra_args`` (e.g. spot coordinates for :class:`SpatialProposal`) are
    forwarded verbatim to both ``sample_and_log_q`` and ``log_q``.
    """
    sample_keys = jax.random.split(key, n_samples)
    samples, log_q_samples = jax.vmap(lambda k: model.sample_and_log_q(k, X, always_passive, context, graph, *extra_args))(sample_keys)
    log_q_true = model.log_q(true_value, X, always_passive, graph, *extra_args)
    free = ~always_passive
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0) + 1e-6
    rank = jnp.mean((log_q_samples <= log_q_true).astype(jnp.float32))  # fraction of samples the truth out-densities
    return (
        np.asarray(true_value[free]),
        np.asarray(mean[free]),
        np.asarray(std[free]),
        float(rank),
        np.asarray(free),
    )


def summarize_coverage(truth, mean, std, level):
    """Wrap ``empirical_coverage`` and report a single scalar mean coverage."""
    if truth.shape[0] == 0:
        return None
    report = empirical_coverage(jnp.asarray(truth), jnp.asarray(mean), jnp.asarray(std), level)
    return {"coverage": [float(v) for v in np.atleast_1d(report.coverage)], "mean_interval_width": [float(v) for v in np.atleast_1d(report.mean_interval_width)], "n_observations": int(report.n_observations)}


def rank_summary(ranks: list[float]) -> dict:
    """Summarize SBC rank fractions: mean (should be ~0.5) and tail excess."""
    ranks = np.asarray(ranks)
    if ranks.size == 0:
        return {"n": 0}
    return {
        "n": int(ranks.size),
        "mean_rank": float(np.mean(ranks)),
        "frac_below_0.1": float(np.mean(ranks < 0.1)),
        "frac_above_0.9": float(np.mean(ranks > 0.9)),
    }


def main():
    """Run the calibration harness and print/save its report."""
    args = parse_args()
    graphs, F_values, pathway_graph = load_shard(args.data)
    n_examples = min(args.n_examples, len(graphs))
    n_genes = F_values.shape[1]
    H = F_values.shape[2]
    observation_dim = graphs[0].nodes["X"].shape[-1]

    key = jax.random.PRNGKey(args.seed)
    spatial_key, loading_key, key = jax.random.split(key, 3)
    spatial = SpatialProposal(H, observation_dim, args.hidden_dim, args.depth, key=spatial_key, gat_depth=args.gat_depth, base=args.spatial_base, degrees_of_freedom=args.degrees_of_freedom)
    loading = LoadingProposal(H, n_genes, args.loading_hidden_dim, args.depth, key=loading_key, gat_depth=args.gat_depth, base=args.loading_base, degrees_of_freedom=args.degrees_of_freedom)
    if args.spatial_checkpoint is not None:
        import equinox as eqx
        spatial = eqx.tree_deserialise_leaves(args.spatial_checkpoint, spatial)
    if args.loading_checkpoint is not None:
        import equinox as eqx
        loading = eqx.tree_deserialise_leaves(args.loading_checkpoint, loading)

    results = {}
    for mode, frac in (("global", 0.0), ("blocked", args.context_frac)):
        buckets = {
            "L_all": {"truth": [], "mean": [], "std": [], "ranks": []},
            "L_boundary": {"truth": [], "mean": [], "std": [], "ranks": []},
            "L_interior": {"truth": [], "mean": [], "std": [], "ranks": []},
            "F_all": {"truth": [], "mean": [], "std": [], "ranks": []},
        }
        for index in range(n_examples):
            graph = graphs[index]
            nodes = graph.nodes
            model_graph = graph_to_model_graph(graph)
            n_spots = nodes["L"].shape[0]
            key, l_key, f_key, mask_l_key, mask_f_key = jax.random.split(key, 5)

            always_passive_l = jax.random.uniform(mask_l_key, (n_spots,)) < frac
            truth_l, mean_l, std_l, rank_l, free_l = evaluate_head(l_key, spatial, nodes["X"], nodes["L"], always_passive_l, nodes["L"], model_graph, args.n_samples, nodes["coords"])
            buckets["L_all"]["truth"].append(truth_l); buckets["L_all"]["mean"].append(mean_l); buckets["L_all"]["std"].append(std_l); buckets["L_all"]["ranks"].append(rank_l)

            scores = np.asarray(boundary_score(nodes["L"], model_graph))
            is_boundary = scores >= np.median(scores)
            free_positions = np.where(np.asarray(free_l))[0]
            stratum_boundary = is_boundary[free_positions]
            if stratum_boundary.any():
                buckets["L_boundary"]["truth"].append(truth_l[stratum_boundary]); buckets["L_boundary"]["mean"].append(mean_l[stratum_boundary]); buckets["L_boundary"]["std"].append(std_l[stratum_boundary])
            if (~stratum_boundary).any():
                buckets["L_interior"]["truth"].append(truth_l[~stratum_boundary]); buckets["L_interior"]["mean"].append(mean_l[~stratum_boundary]); buckets["L_interior"]["std"].append(std_l[~stratum_boundary])

            always_passive_f = jax.random.uniform(mask_f_key, (n_genes,)) < frac
            F_true = F_values[index]
            truth_f, mean_f, std_f, rank_f, _ = evaluate_head(f_key, loading, nodes["X"], F_true, always_passive_f, F_true, pathway_graph, args.n_samples)
            buckets["F_all"]["truth"].append(truth_f); buckets["F_all"]["mean"].append(mean_f); buckets["F_all"]["std"].append(std_f); buckets["F_all"]["ranks"].append(rank_f)

        mode_report = {}
        for name, bucket in buckets.items():
            truth = np.concatenate(bucket["truth"], axis=0) if bucket["truth"] else np.zeros((0, H))
            mean = np.concatenate(bucket["mean"], axis=0) if bucket["mean"] else np.zeros((0, H))
            std = np.concatenate(bucket["std"], axis=0) if bucket["std"] else np.zeros((0, H))
            entry = {"coverage": summarize_coverage(truth, mean, std, args.coverage_level)}
            if bucket["ranks"]:
                entry["sbc_rank"] = rank_summary(bucket["ranks"])
            mode_report[name] = entry
        results[mode] = mode_report

    report = {"n_examples": n_examples, "n_samples": args.n_samples, "context_frac": args.context_frac, "coverage_level": args.coverage_level, "results": results}
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
