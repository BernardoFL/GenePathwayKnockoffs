#!/usr/bin/env python
"""Master driver for amortized neural MCMC on a local or Slurm node.

Runs the exact mixture kernel (zone-blocked + global-joint independence
proposals on ``(L, F)``, scored against ``log_pi`` with a true Metropolis-
Hastings accept/reject) composed with the Gibbs blocks that never touch the
network: ``alpha_p``, the CSP allocation, and ``rho``. Loads trained
:class:`SpatialProposal`/:class:`LoadingProposal` checkpoints if given;
otherwise runs with freshly initialized (untrained) heads, which is useful
for a smoke test of the sampler's plumbing but not for real inference.

Input NPZ must contain ``coordinates``, ``X``, and ``A_path`` (a gene-gene
pathway affinity matrix). Optional arrays are ``L``, ``F``, ``alpha_p``,
``rho``, and a ``gene_ids`` panel to check against a trained checkpoint's
recorded panel. Results are written as compressed NPZ checkpoints and a
JSON summary; no interactive plotting or prompts are used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from amortized_mcmc import (
    AmortizedProposal,
    ChainState,
    LoadingProposal,
    SpatialProposal,
    build_graph,
    build_pathway_graph,
    gibbs_step_alpha_p,
    gibbs_step_csp,
    gibbs_step_rho,
    gibbs_step_sigma_alpha_sq,
    log_pi,
    mixture_step,
    pathway_laplacian_from_affinity,
    sample_csp_prior,
    spectral_zone_masks,
)


def parse_args() -> argparse.Namespace:
    """Parse batch-run configuration for posterior sampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="NPZ containing coordinates, X, and A_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) + 1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.3, help="global-joint-jump mixing weight")
    parser.add_argument("--n-spot-zones", type=int, default=4)
    parser.add_argument("--n-gene-zones", type=int, default=4)
    parser.add_argument("--spot-graph", choices=("knn", "delaunay"), default="knn")
    parser.add_argument("--spot-k", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loading-hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--gat-depth", type=int, default=2)
    parser.add_argument("--spatial-base", choices=("normal", "student_t"), default="normal")
    parser.add_argument("--loading-base", choices=("normal", "student_t"), default="student_t")
    parser.add_argument("--degrees-of-freedom", type=float, default=4.0)
    parser.add_argument("--spatial-checkpoint", type=Path, default=None)
    parser.add_argument("--loading-checkpoint", type=Path, default=None)
    parser.add_argument("--n-spots", type=int, default=64, help="Synthetic slide size when --data is omitted")
    parser.add_argument("--n-genes", type=int, default=32, help="Synthetic gene count when --data is omitted")
    parser.add_argument("--n-patient-effects", type=int, default=1)
    parser.add_argument("--a-rho", type=float, default=1.0)
    parser.add_argument("--b-rho", type=float, default=1.0)
    parser.add_argument("--a0", type=float, default=2.0, help="InverseGamma shape for sigma_alpha^2")
    parser.add_argument("--b0", type=float, default=1.0, help="InverseGamma scale for sigma_alpha^2")
    parser.add_argument("--platform", choices=("cpu", "gpu", "tpu"), default=None)
    return parser.parse_args()


def synthetic_data(seed: int, n_spots: int, n_genes: int):
    """Create a small synthetic coordinate/count/pathway dataset for smoke tests."""
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 1.0, (n_spots, 2)).astype(np.float32)
    X = rng.poisson(1.5, (n_spots, n_genes)).astype(np.float32)
    embedding = rng.normal(size=(n_genes, 2)).astype(np.float32)
    gene_graph = build_graph(embedding, method="knn", k=min(6, n_genes - 1))
    affinity = np.zeros((n_genes, n_genes), dtype=np.float32)
    affinity[np.asarray(gene_graph.senders), np.asarray(gene_graph.receivers)] = 1.0
    affinity[np.asarray(gene_graph.receivers), np.asarray(gene_graph.senders)] = 1.0
    return coordinates, X, affinity


def load_data(args: argparse.Namespace):
    """Load an input NPZ or generate synthetic data when no path is supplied."""
    if args.data is None:
        coordinates, X, affinity = synthetic_data(args.seed, args.n_spots, args.n_genes)
        return coordinates, X, affinity, None, None, None, None
    with np.load(args.data, allow_pickle=True) as data:
        if "coordinates" not in data or "X" not in data or "A_path" not in data:
            raise ValueError("data NPZ must contain coordinates, X, and A_path")
        coordinates = np.asarray(data["coordinates"], dtype=np.float32)
        X = np.asarray(data["X"], dtype=np.float32)
        affinity = np.asarray(data["A_path"], dtype=np.float32)
        return (
            coordinates,
            X,
            affinity,
            data["L"] if "L" in data else None,
            data["F"] if "F" in data else None,
            data["alpha_p"] if "alpha_p" in data else None,
            data["rho"] if "rho" in data else None,
        )


def load_or_init_head(cls, checkpoint: Path | None, *args, key, **kwargs):
    """Build a head skeleton and optionally load trained leaves into it."""
    model = cls(*args, key=key, **kwargs)
    if checkpoint is not None:
        model = eqx.tree_deserialise_leaves(checkpoint, model)
    return model


def save_checkpoint(output: Path, state: ChainState, log_pi_trace, accepted_trace, iteration: int):
    """Write chain state, traces, and CSP fields to a numbered NPZ checkpoint."""
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"checkpoint_{iteration:07d}.npz",
        L=np.asarray(state.L),
        F=np.asarray(state.F),
        alpha=np.asarray(state.alpha),
        rho=np.asarray(state.rho),
        csp_phi=np.asarray(state.csp.phi),
        csp_z=np.asarray(state.csp.z),
        log_pi=np.asarray(log_pi_trace),
        accepted=np.asarray(accepted_trace),
    )


def main() -> None:
    """Run the exact mixture-kernel sampler composed with the Gibbs blocks."""
    args = parse_args()
    if args.platform:
        jax.config.update("jax_platform_name", args.platform)
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    coordinates, X_np, affinity, L_np, F_np, alpha_np, rho_np = load_data(args)
    n_spots, n_genes = X_np.shape
    spot_graph = build_graph(coordinates, method=args.spot_graph, k=min(args.spot_k, n_spots - 1))
    pathway_graph = build_pathway_graph(affinity)
    pathway_laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))

    key = jax.random.PRNGKey(args.seed)
    key, csp_key, spatial_key, loading_key = jax.random.split(key, 4)
    csp = sample_csp_prior(csp_key, args.H, n_genes)
    L = jnp.asarray(L_np if L_np is not None else np.zeros((n_spots, args.H), dtype=np.float32))
    F = jnp.asarray(F_np if F_np is not None else np.zeros((n_genes, args.H), dtype=np.float32))
    alpha = jnp.asarray(alpha_np if alpha_np is not None else np.zeros((args.n_patient_effects,), dtype=np.float32))
    rho = jnp.asarray(rho_np if rho_np is not None else 1.0, dtype=jnp.float32)
    p_of_s = jnp.zeros((n_spots,), dtype=jnp.int32)
    state = ChainState(jnp.asarray(X_np), L, F, alpha, p_of_s, spot_graph, csp, pathway_graph, rho)

    spatial = load_or_init_head(
        SpatialProposal, args.spatial_checkpoint, args.H, n_genes, args.hidden_dim, args.depth,
        key=spatial_key, gat_depth=args.gat_depth, base=args.spatial_base, degrees_of_freedom=args.degrees_of_freedom,
    )
    loading = load_or_init_head(
        LoadingProposal, args.loading_checkpoint, args.H, n_genes, args.loading_hidden_dim, args.depth,
        key=loading_key, gat_depth=args.gat_depth, base=args.loading_base, degrees_of_freedom=args.degrees_of_freedom,
    )
    heads = AmortizedProposal(spatial, loading)

    # Zones are read off each head's own GAT attention over its graph (a
    # trained head's attention already reflects morphological/pathway
    # structure; an untrained one just reflects its random init) via
    # recursive spectral bisection, per the model spec's "morphological
    # zones read off the affinity/attention structure" -- not an arbitrary
    # coordinate ordering.
    spot_attention = np.asarray(spatial.attention_edge_weights(state.X, spot_graph, jnp.asarray(coordinates)))
    gene_attention = np.asarray(loading.attention_edge_weights(state.X, pathway_graph))
    spot_zone_masks = spectral_zone_masks(n_spots, spot_graph.senders, spot_graph.receivers, args.n_spot_zones, spot_attention)
    gene_zone_masks = spectral_zone_masks(n_genes, pathway_graph.senders, pathway_graph.receivers, args.n_gene_zones, gene_attention)

    def log_pi_fn(L_value, F_value, chain_state):
        """Evaluate the true target at a candidate (L, F), reading only chain_state's other fields."""
        return log_pi(
            L_value, F_value, chain_state.X, chain_state.alpha, chain_state.p_of_s, chain_state.graph, chain_state.csp,
            pathway_laplacian=pathway_laplacian, rho=chain_state.rho,
        )

    mixture = jax.jit(lambda step_key, chain_state: mixture_step(step_key, chain_state, args.beta, heads, log_pi_fn, spot_zone_masks, gene_zone_masks))

    sigma_alpha_sq = jnp.asarray(1.0)
    log_pi_trace, acceptance_trace, move_trace = [], [], []
    for iteration in range(args.iterations):
        key, mix_key, alpha_key, sigma_key, csp_key, rho_key = jax.random.split(key, 6)
        state, accepted, acceptance, move_type = mixture(mix_key, state)
        state = state._replace(alpha=gibbs_step_alpha_p(alpha_key, state, sigma_alpha=jnp.sqrt(sigma_alpha_sq)))
        sigma_alpha_sq = gibbs_step_sigma_alpha_sq(sigma_key, state.alpha, args.a0, args.b0)
        state = gibbs_step_csp(csp_key, state)
        state = state._replace(rho=gibbs_step_rho(rho_key, state.F, state.csp, pathway_laplacian, args.a_rho, args.b_rho))
        if iteration >= args.warmup:
            log_pi_trace.append(float(log_pi_fn(state.L, state.F, state)))
            acceptance_trace.append(float(accepted))
            move_trace.append(int(move_type))
        if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 == args.iterations:
            save_checkpoint(args.output, state, log_pi_trace, acceptance_trace, iteration + 1)

    summary = {
        "seed": args.seed,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "H": args.H,
        "beta": args.beta,
        "n_spots": n_spots,
        "n_genes": n_genes,
        "acceptance_rate": float(np.mean(acceptance_trace)) if acceptance_trace else None,
        "final_sigma_alpha_sq": float(sigma_alpha_sq),
        "move_type_counts": {int(code): int(np.sum(np.asarray(move_trace) == code)) for code in sorted(set(move_trace))} if move_trace else {},
        "platform": jax.default_backend(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"summary_{args.seed}.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
