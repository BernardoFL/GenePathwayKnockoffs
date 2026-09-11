#!/usr/bin/env python
"""Master driver for amortized neural MCMC on a local or Slurm node.

Input NPZ files must contain ``coordinates`` and ``X``. Optional arrays are
``p_of_s``, ``L``, ``F``, and ``alpha``. Results are written as compressed NPZ
checkpoints and a JSON summary; no interactive plotting or prompts are used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from amortized_mcmc import (
    ChainState,
    LoadingProposal,
    SpatialProposal,
    build_graph,
    blackjax_mh_step,
    gibbs_step_alpha,
    gibbs_step_csp,
    log_pi_L,
    sample_csp_prior,
)


def parse_args() -> argparse.Namespace:
    """Parse batch-run configuration for posterior sampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="NPZ containing coordinates and X")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) + 1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--graph", choices=("knn", "delaunay"), default="knn")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--gat-depth", type=int, default=2)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--n-spots", type=int, default=64, help="Synthetic slide size when --data is omitted")
    parser.add_argument("--n-genes", type=int, default=32, help="Synthetic gene count when --data is omitted")
    parser.add_argument("--n-patient-effects", type=int, default=1)
    parser.add_argument("--platform", choices=("cpu", "gpu", "tpu"), default=None)
    return parser.parse_args()


def synthetic_data(seed: int, n_spots: int, n_genes: int):
    """Create a small synthetic coordinate/count dataset for smoke tests."""
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 1.0, (n_spots, 2)).astype(np.float32)
    X = rng.poisson(1.5, (n_spots, n_genes)).astype(np.float32)
    return coordinates, X


def load_data(args: argparse.Namespace):
    """Load an input NPZ or generate synthetic data when no path is supplied."""
    if args.data is None:
        coordinates, X = synthetic_data(args.seed, args.n_spots, args.n_genes)
        return coordinates, X, None, None, None
    with np.load(args.data) as data:
        if "coordinates" not in data or "X" not in data:
            raise ValueError("data NPZ must contain coordinates and X")
        coordinates = np.asarray(data["coordinates"], dtype=np.float32)
        X = np.asarray(data["X"], dtype=np.float32)
        return (
            coordinates,
            X,
            data["L"] if "L" in data else None,
            data["F"] if "F" in data else None,
            data["alpha"] if "alpha" in data else None,
        )


def save_checkpoint(output: Path, state: ChainState, l_trace, accepted, iteration: int):
    """Write chain state, traces, and CSP fields to a numbered NPZ checkpoint."""
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"checkpoint_{iteration:07d}.npz",
        L=np.asarray(state.L),
        F=np.asarray(state.F),
        alpha=np.asarray(state.alpha),
        csp_phi=np.asarray(state.csp.phi) if state.csp is not None else None,
        csp_z=np.asarray(state.csp.z) if state.csp is not None else None,
        log_pi_L=np.asarray(l_trace),
        accepted=np.asarray(accepted),
    )


def main() -> None:
    """Run BlackJAX L updates and separate alpha/CSP updates on a batch node."""
    args = parse_args()
    if args.platform:
        jax.config.update("jax_platform_name", args.platform)
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup cannot be negative")
    coordinates, X_np, L_np, F_np, alpha_np = load_data(args)
    n_spots, n_genes = X_np.shape
    graph = build_graph(coordinates, method=args.graph, k=args.k)
    key = jax.random.PRNGKey(args.seed)
    key, csp_key, spatial_key, loading_key = jax.random.split(key, 4)
    csp = sample_csp_prior(csp_key, args.H, n_genes)
    L = jnp.asarray(L_np if L_np is not None else np.zeros((n_spots, args.H), dtype=np.float32))
    F = jnp.asarray(F_np if F_np is not None else np.zeros((n_genes, args.H), dtype=np.float32))
    alpha = jnp.asarray(alpha_np if alpha_np is not None else np.zeros((args.n_patient_effects,), dtype=np.float32))
    p_of_s = jnp.zeros((n_spots,), dtype=jnp.int32)
    state = ChainState(jnp.asarray(X_np), L, F, alpha, p_of_s, graph, csp)
    spatial = SpatialProposal(args.H, n_genes, args.hidden_dim, args.gat_depth, key=spatial_key)
    # Construct the independent F head now so checkpoints can be extended with
    # F proposals without changing the job's model initialization contract.
    _ = LoadingProposal(n_genes, n_genes, args.H, key=loading_key)

    l_trace = []
    acceptance_trace = []
    for iteration in range(args.iterations):
        key, l_key, alpha_key, csp_key = jax.random.split(key, 4)
        # Rebuild the target closure after every Gibbs update so the exact L
        # target uses the current patient effects and loading state.
        target = lambda l_value: log_pi_L(
            l_value, state.X, state.F, state.alpha, state.p_of_s, state.graph
        )
        state, accepted, acceptance = blackjax_mh_step(l_key, state, spatial, target)
        state = state._replace(alpha=gibbs_step_alpha(alpha_key, state))
        state = gibbs_step_csp(csp_key, state)
        if iteration >= args.warmup:
            l_trace.append(float(log_pi_L(state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph)))
            acceptance_trace.append(float(accepted))
        if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 == args.iterations:
            save_checkpoint(args.output, state, l_trace, acceptance_trace, iteration + 1)
    summary = {
        "seed": args.seed,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "H": args.H,
        "graph": args.graph,
        "n_spots": n_spots,
        "n_genes": n_genes,
        "acceptance_rate": float(np.mean(acceptance_trace)) if acceptance_trace else None,
        "platform": jax.default_backend(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"summary_{args.seed}.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()