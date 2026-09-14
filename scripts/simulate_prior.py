#!/usr/bin/env python
"""Prior-only training-data simulator for the amortized heads (spec §6).

Every example is a single NumPyro ``Predictive`` prior draw of the complete
generative model (CSP hyperpriors, allocations, Horseshoe/pathway-coupled
``F``, MRF ``L``, patient effects, Poisson ``X``). That one ``(L, X)`` /
``(F, X)`` pair *is* a posterior sample at that ``X`` -- no reference MCMC
chain runs here, and no transition pairs are recorded. Spot count varies
per example; node arrays are flattened and delimited by ``n_node``/``n_edge``
in the jraph flattened-graph convention. The gene panel (``G``) and CSP
truncation (``H``) are fixed across the whole shard, matching a real slide's
post-QC gene set and the trained heads' output width.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive

from amortized_mcmc import build_graph, full_generative_model, pathway_laplacian_from_affinity


def parse_args() -> argparse.Namespace:
    """Parse simulator settings for one training-data shard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--min-spots", type=int, default=32)
    parser.add_argument("--max-spots", type=int, default=128)
    parser.add_argument("--n-genes", type=int, default=32)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--n-patient-effects", type=int, default=1)
    parser.add_argument("--spot-graph", choices=("knn", "delaunay"), default="knn")
    parser.add_argument("--spot-k", type=int, default=6)
    parser.add_argument("--pathway-k", type=int, default=6, help="k-NN degree for the synthetic gene pathway graph")
    parser.add_argument("--a-rho", type=float, default=1.0)
    parser.add_argument("--b-rho", type=float, default=1.0)
    return parser.parse_args()


def build_synthetic_pathway_affinity(rng: np.random.Generator, n_genes: int, k: int) -> np.ndarray:
    """Build a fixed, shared synthetic gene-gene pathway affinity matrix.

    This stands in for the real annotation graph ``A_path`` over the
    dataset's post-QC gene panel; production usage should load that graph
    instead. Its edges are read off a k-NN graph over a random gene
    embedding purely to get a connected, sparse, non-trivial topology --
    a placeholder, not a claim about real pathway structure. Whatever the
    source, this graph must stay a distinct object from any data-driven
    co-expression graph the knockoff filter later tests (see
    :func:`amortized_mcmc.assert_graphs_not_aliased`).
    """
    embedding = rng.normal(size=(n_genes, 2)).astype(np.float32)
    gene_graph = build_graph(embedding, method="knn", k=min(k, n_genes - 1))
    affinity = np.zeros((n_genes, n_genes), dtype=np.float32)
    senders = np.asarray(gene_graph.senders)
    receivers = np.asarray(gene_graph.receivers)
    affinity[senders, receivers] = 1.0
    affinity[receivers, senders] = 1.0
    return affinity


def main() -> None:
    """Draw ``n_examples`` independent prior states and write one npz shard."""
    args = parse_args()
    if args.n_examples < 1:
        raise ValueError("n-examples must be positive")
    if not (0 < args.min_spots <= args.max_spots):
        raise ValueError("min-spots must be positive and at most max-spots")
    if args.n_genes < 2:
        raise ValueError("n-genes must be at least 2 for a pathway graph")

    rng = np.random.default_rng(args.seed)
    gene_ids = np.array([f"GENE_{index:05d}" for index in range(args.n_genes)])
    affinity = build_synthetic_pathway_affinity(rng, args.n_genes, args.pathway_k)
    pathway_laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_examples)
    flat_coords, flat_X, flat_L = [], [], []
    edge_index_chunks, n_node, n_edge = [], [], []
    F_values, rho_values, alpha_p_values, sigma_alpha_sq_values = [], [], [], []
    csp_v, csp_z, csp_phi, csp_nu, csp_alpha_csp, csp_b_phi = [], [], [], [], [], []

    node_offset = 0
    for example_index, key in enumerate(keys):
        n_spots = int(rng.integers(args.min_spots, args.max_spots + 1))
        coordinates = rng.uniform(0.0, 1.0, (n_spots, 2)).astype(np.float32)
        spot_graph = build_graph(coordinates, method=args.spot_graph, k=min(args.spot_k, n_spots - 1))

        predictive = Predictive(full_generative_model, num_samples=1)
        draw = predictive(
            key,
            n_spots=n_spots,
            n_genes=args.n_genes,
            H=args.H,
            senders=spot_graph.senders,
            receivers=spot_graph.receivers,
            n_patient_effects=args.n_patient_effects,
            pathway_laplacian=pathway_laplacian,
            a_rho=args.a_rho,
            b_rho=args.b_rho,
        )
        draw = {name: np.asarray(value[0]) for name, value in draw.items()}

        flat_coords.append(coordinates)
        flat_X.append(draw["X"])
        flat_L.append(draw["L"])
        edges = np.stack((np.asarray(spot_graph.senders), np.asarray(spot_graph.receivers)), axis=0)
        edge_index_chunks.append(edges + node_offset)
        n_node.append(n_spots)
        n_edge.append(spot_graph.n_edges)
        node_offset += n_spots

        F_values.append(draw["F"])
        rho_values.append(draw["rho"])
        alpha_p_values.append(draw["alpha_p"])
        sigma_alpha_sq_values.append(draw["sigma_alpha_sq"])
        csp_v.append(draw["v"])
        csp_z.append(np.stack([draw[f"z_{h}"] for h in range(args.H)]))
        csp_phi.append(draw["phi"])
        csp_nu.append(draw["lambda_local"])
        csp_alpha_csp.append(draw["alpha_csp"])
        csp_b_phi.append(draw["b_phi"])

        if (example_index + 1) % max(1, args.n_examples // 20) == 0:
            print(f"drew {example_index + 1}/{args.n_examples} prior states", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        n_node=np.asarray(n_node, dtype=np.int32),
        n_edge=np.asarray(n_edge, dtype=np.int32),
        coords=np.concatenate(flat_coords, axis=0).astype(np.float32),
        edge_index=np.concatenate(edge_index_chunks, axis=1).astype(np.int32),
        X=np.concatenate(flat_X, axis=0).astype(np.float32),
        L=np.concatenate(flat_L, axis=0).astype(np.float32),
        F=np.asarray(F_values, dtype=np.float32),
        A_path=affinity,
        rho=np.asarray(rho_values, dtype=np.float32),
        alpha_p=np.asarray(alpha_p_values, dtype=np.float32),
        sigma_alpha_sq=np.asarray(sigma_alpha_sq_values, dtype=np.float32),
        csp_v=np.asarray(csp_v, dtype=np.float32),
        csp_z=np.asarray(csp_z, dtype=np.int32),
        csp_phi=np.asarray(csp_phi, dtype=np.float32),
        csp_nu=np.asarray(csp_nu, dtype=np.float32),
        csp_alpha_csp=np.asarray(csp_alpha_csp, dtype=np.float32),
        csp_b_phi=np.asarray(csp_b_phi, dtype=np.float32),
        gene_ids=gene_ids,
        H=np.asarray(args.H, dtype=np.int32),
    )
    print(f"wrote {args.n_examples} prior draws to {args.output}")


if __name__ == "__main__":
    main()
