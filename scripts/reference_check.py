#!/usr/bin/env python
"""Cross-check the exact mixture-kernel sampler against NumPyro (spec §10.2/§10.5).

On one small synthetic slide:

1. Draw a true prior state and observation ``X`` from ``full_generative_model``.
2. Run NumPyro's ``DiscreteHMCGibbs(NUTS(full_generative_model))`` conditioned
   on that ``X`` -- NUTS handles the continuous block (``L, F, alpha_p, v,
   phi_slab, rho, sigma_alpha_sq``), Gibbs enumerates the discrete ``z_h``
   allocations -- as the slow-but-correct reference posterior. This chain
   never feeds training; it is validation only.
3. Quickly warm-start both amortized heads on freshly simulated prior draws
   of matching shape (a light in-script stand-in for a real
   ``scripts/train_amortizer.py`` run -- pass ``--spatial-checkpoint`` /
   ``--loading-checkpoint`` to use real trained heads instead).
4. Run our own exact ``mixture_step`` + Gibbs (``alpha_p``, CSP, ``rho``,
   ``sigma_alpha_sq``) chain conditioned on the same ``X``.
5. Compare posterior summaries (mean, std, effective sample size) for a few
   scalar functionals between the two chains and report a PASS/WARN
   z-score-style check per functional.

This is a distributional smoke test on a tiny slide, not a proof of exact
equivalence -- treat the report as a diagnostic, and expect noisier
agreement the less the amortizer has actually been trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, Predictive

from amortized_mcmc import (
    AmortizedProposal,
    ChainState,
    LoadingProposal,
    SpatialProposal,
    build_graph,
    build_pathway_graph,
    effective_sample_size,
    full_generative_model,
    gibbs_step_alpha_p,
    gibbs_step_csp,
    gibbs_step_rho,
    gibbs_step_sigma_alpha_sq,
    log_pi,
    mixture_step,
    pathway_laplacian_from_affinity,
    spectral_zone_masks,
)
from amortized_mcmc.csp import CSPState


def parse_args() -> argparse.Namespace:
    """Parse small-slide cross-check settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-spots", type=int, default=12)
    parser.add_argument("--n-genes", type=int, default=6)
    parser.add_argument("--H", type=int, default=3)
    parser.add_argument("--spot-k", type=int, default=4)
    parser.add_argument("--pathway-k", type=int, default=3)
    parser.add_argument("--train-examples", type=int, default=24, help="matching-shape prior draws for the in-script amortizer warm-start")
    parser.add_argument("--train-epochs", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--loading-hidden-dim", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--spatial-checkpoint", type=Path, default=None)
    parser.add_argument("--loading-checkpoint", type=Path, default=None)
    parser.add_argument("--reference-warmup", type=int, default=500)
    parser.add_argument("--reference-samples", type=int, default=1000)
    parser.add_argument("--our-iterations", type=int, default=2000)
    parser.add_argument("--our-warmup", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--n-spot-zones", type=int, default=2)
    parser.add_argument("--n-gene-zones", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_graphs(args, rng):
    """Build the fixed spot and pathway graphs for this slide."""
    coordinates = rng.uniform(0.0, 1.0, (args.n_spots, 2)).astype(np.float32)
    spot_graph = build_graph(coordinates, method="knn", k=min(args.spot_k, args.n_spots - 1))
    gene_embedding = rng.normal(size=(args.n_genes, 2)).astype(np.float32)
    gene_graph = build_graph(gene_embedding, method="knn", k=min(args.pathway_k, args.n_genes - 1))
    affinity = np.zeros((args.n_genes, args.n_genes), dtype=np.float32)
    affinity[np.asarray(gene_graph.senders), np.asarray(gene_graph.receivers)] = 1.0
    affinity[np.asarray(gene_graph.receivers), np.asarray(gene_graph.senders)] = 1.0
    pathway_graph = build_pathway_graph(affinity)
    pathway_laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    return coordinates, spot_graph, pathway_graph, pathway_laplacian


def model_kwargs(args, spot_graph, pathway_laplacian):
    """Static NumPyro model arguments shared by prior draws and the reference chain."""
    return dict(
        n_spots=args.n_spots, n_genes=args.n_genes, H=args.H,
        senders=spot_graph.senders, receivers=spot_graph.receivers,
        pathway_laplacian=pathway_laplacian,
    )


def draw_representative_prior(key, args, spot_graph, pathway_laplacian, max_observed_count: int = 30, max_attempts: int = 50):
    """Reject pathologically extreme prior draws before using one as the slide.

    The Horseshoe local scale is Half-Cauchy and genuinely heavy-tailed by
    design (per the model spec), so an unlucky prior draw can occasionally
    produce enormous ``F`` entries and Poisson counts in the hundreds or
    thousands. Such a draw makes the resulting posterior astronomically
    peaked, so *no* independence proposal -- however well trained -- has a
    realistic chance of matching it by chance; that would make this
    validation script's cross-check report a training-capacity ceiling
    that has nothing to do with the sampler's own correctness. Rerolling
    for a slide whose observed counts stay in a plausible range keeps the
    comparison meaningful.
    """
    for attempt in range(max_attempts):
        draw_key = jax.random.fold_in(key, attempt)
        draw = Predictive(full_generative_model, num_samples=1)(draw_key, n_patient_effects=1, **model_kwargs(args, spot_graph, pathway_laplacian))
        draw = {name: np.asarray(value[0]) for name, value in draw.items()}
        if draw["X"].max() <= max_observed_count:
            if attempt > 0:
                print(f"drew a representative slide after {attempt + 1} attempts (X max = {draw['X'].max():.0f})", flush=True)
            return draw
    raise RuntimeError(f"no prior draw with max observed count <= {max_observed_count} found in {max_attempts} attempts")


def warm_start_heads(key, args, spot_graph, coordinates, pathway_graph, pathway_laplacian):
    """Quickly fit both heads on freshly simulated matching-shape prior draws.

    Every example reuses the *same* spot graph (``spot_graph``, the one the
    query slide itself uses) and only draws a fresh ``(X, L, F)`` -- a k-NN
    graph's edge count varies example-to-example with a fresh random point
    cloud (mutual-neighbor pairs get deduplicated unevenly), which would
    otherwise make stacking examples into a single batch unsafe. Reusing one
    graph lets the whole batch be stacked and the per-epoch update run
    through a single ``jax.vmap`` + ``eqx.filter_jit`` step instead of a
    slow, un-jitted Python loop over examples (which otherwise dominates
    wall time here).
    """
    rng = np.random.default_rng(args.seed + 1)
    X_list, L_list, F_list = [], [], []
    for _ in range(args.train_examples):
        key, draw_key = jax.random.split(key)
        draw = Predictive(full_generative_model, num_samples=1)(
            draw_key, n_patient_effects=1, **model_kwargs(args, spot_graph, pathway_laplacian)
        )
        draw = {name: np.asarray(value[0]) for name, value in draw.items()}
        X_list.append(draw["X"]); L_list.append(draw["L"]); F_list.append(draw["F"])

    X_batch = jnp.asarray(np.stack(X_list))
    L_batch = jnp.asarray(np.stack(L_list))
    F_batch = jnp.asarray(np.stack(F_list))

    spatial_key, loading_key, key = jax.random.split(key, 3)
    spatial = SpatialProposal(args.H, args.n_genes, args.hidden_dim, args.depth, key=spatial_key)
    loading = LoadingProposal(args.H, args.n_genes, args.loading_hidden_dim, args.depth, key=loading_key)
    if args.spatial_checkpoint is not None:
        spatial = eqx.tree_deserialise_leaves(args.spatial_checkpoint, spatial)
    if args.loading_checkpoint is not None:
        loading = eqx.tree_deserialise_leaves(args.loading_checkpoint, loading)
        return spatial, loading

    def spatial_loss_fn(model, mask_keys):
        """Average negative log-density of the true L under random context masks."""
        def per_example(X, L, mask_key):
            """Score one example's true L under a fresh random context mask."""
            always_passive = jax.random.uniform(mask_key, (args.n_spots,)) < 0.4
            return -model.log_q(L, X, always_passive, spot_graph, coordinates)

        return jnp.mean(jax.vmap(per_example)(X_batch, L_batch, mask_keys))

    def loading_loss_fn(model, mask_keys):
        """Average negative log-density of the true F under random context masks."""
        def per_example(X, F, mask_key):
            """Score one example's true F under a fresh random context mask."""
            always_passive = jax.random.uniform(mask_key, (args.n_genes,)) < 0.4
            return -model.log_q(F, X, always_passive, pathway_graph)

        return jnp.mean(jax.vmap(per_example)(X_batch, F_batch, mask_keys))

    @eqx.filter_jit
    def train_step(spatial, loading, spatial_state, loading_state, step_key):
        """Run one jitted gradient step for both heads over the whole batch."""
        spatial_keys, loading_keys = jax.random.split(step_key)
        spatial_keys = jax.random.split(spatial_keys, args.train_examples)
        loading_keys = jax.random.split(loading_keys, args.train_examples)
        grads_spatial = eqx.filter_grad(spatial_loss_fn)(spatial, spatial_keys)
        grads_loading = eqx.filter_grad(loading_loss_fn)(loading, loading_keys)
        updates, spatial_state = optimizer.update(grads_spatial, spatial_state, spatial)
        spatial = eqx.apply_updates(spatial, updates)
        updates, loading_state = optimizer.update(grads_loading, loading_state, loading)
        loading = eqx.apply_updates(loading, updates)
        return spatial, loading, spatial_state, loading_state

    optimizer = optax.adam(1e-3)
    spatial_state = optimizer.init(eqx.filter(spatial, eqx.is_array))
    loading_state = optimizer.init(eqx.filter(loading, eqx.is_array))
    for _ in range(args.train_epochs):
        key, step_key = jax.random.split(key)
        spatial, loading, spatial_state, loading_state = train_step(spatial, loading, spatial_state, loading_state, step_key)
    return spatial, loading


def run_reference_chain(key, args, prior, spot_graph, pathway_laplacian):
    """Run NUTS+discrete-Gibbs on the true generative model, conditioned on X."""
    kernel = DiscreteHMCGibbs(NUTS(full_generative_model))
    mcmc = MCMC(kernel, num_warmup=args.reference_warmup, num_samples=args.reference_samples, num_chains=1, progress_bar=False)
    init_params = {name: prior[name] for name in ("alpha_csp", "b_phi", "v", "phi_slab", "lambda_local", "L", "alpha_p", "rho", "sigma_alpha_sq")}
    init_params["F_by_column"] = prior["F"].T
    init_params.update({f"z_{h}": prior[f"z_{h}"] for h in range(args.H)})
    mcmc.run(key, init_params=init_params, n_patient_effects=1, X=prior["X"], **model_kwargs(args, spot_graph, pathway_laplacian))
    return mcmc.get_samples()


def run_our_chain(key, args, prior, spot_graph, pathway_graph, pathway_laplacian, spatial, loading):
    """Run the mixture kernel + Gibbs blocks on the same observed X."""
    heads = AmortizedProposal(spatial, loading)
    # CSPState.v always includes the fixed trailing v_H = 1 (see sample_csp_prior
    # / gibbs_step_csp._sample_v); the NumPyro model's "v" site is only the
    # H - 1 free sticks, so it must be padded with that fixed entry here.
    csp = CSPState(
        v=jnp.concatenate((jnp.asarray(prior["v"]), jnp.ones((1,)))),
        z=jnp.stack([jnp.asarray(prior[f"z_{h}"]) for h in range(args.H)]),
        phi=jnp.asarray(prior["phi"]), concentration=jnp.asarray(prior["alpha_csp"]), b_phi=jnp.asarray(prior["b_phi"]),
        lam=jnp.asarray(prior["lambda_local"]), a_phi=2.0, theta_infty=1e4,
    )
    state = ChainState(
        X=jnp.asarray(prior["X"]), L=jnp.zeros((args.n_spots, args.H)), F=jnp.zeros((args.n_genes, args.H)),
        alpha=jnp.asarray(prior["alpha_p"]), p_of_s=jnp.zeros((args.n_spots,), dtype=jnp.int32),
        graph=spot_graph, csp=csp, pathway_graph=pathway_graph, rho=jnp.asarray(prior["rho"]),
    )
    spot_zone_masks = spectral_zone_masks(args.n_spots, spot_graph.senders, spot_graph.receivers, args.n_spot_zones)
    gene_zone_masks = spectral_zone_masks(args.n_genes, pathway_graph.senders, pathway_graph.receivers, args.n_gene_zones)

    def log_pi_fn(L_value, F_value, chain_state):
        """Evaluate the true target at a candidate (L, F)."""
        return log_pi(
            L_value, F_value, chain_state.X, chain_state.alpha, chain_state.p_of_s, chain_state.graph, chain_state.csp,
            pathway_laplacian=pathway_laplacian, rho=chain_state.rho,
        )

    def step(carry, step_key):
        """Advance the mixture kernel and every Gibbs block by one iteration."""
        chain_state, sigma_alpha_sq = carry
        mix_key, alpha_key, sigma_key, csp_key, rho_key = jax.random.split(step_key, 5)
        chain_state, accepted, alpha_mh, move_type = mixture_step(mix_key, chain_state, args.beta, heads, log_pi_fn, spot_zone_masks, gene_zone_masks)
        chain_state = chain_state._replace(alpha=gibbs_step_alpha_p(alpha_key, chain_state, sigma_alpha=jnp.sqrt(sigma_alpha_sq)))
        sigma_alpha_sq = gibbs_step_sigma_alpha_sq(sigma_key, chain_state.alpha)
        chain_state = gibbs_step_csp(csp_key, chain_state)
        chain_state = chain_state._replace(rho=gibbs_step_rho(rho_key, chain_state.F, chain_state.csp, pathway_laplacian))
        output = (chain_state.L, chain_state.F, chain_state.alpha, chain_state.rho, sigma_alpha_sq, accepted)
        return (chain_state, sigma_alpha_sq), output

    # A single jitted lax.scan replaces a Python loop over five separately
    # dispatched (until now un-jitted) Gibbs calls per iteration, which
    # otherwise dominates wall time here.
    scan_step = eqx.filter_jit(lambda carry, keys: jax.lax.scan(step, carry, keys))
    keys = jax.random.split(key, args.our_iterations)
    _, (L_trace, F_trace, alpha_trace, rho_trace, sigma_trace, accepted_trace) = scan_step((state, jnp.asarray(1.0)), keys)

    post_warmup = slice(args.our_warmup, None)
    print(f"our chain acceptance rate (post-warmup): {float(jnp.mean(accepted_trace[post_warmup])):.3f}", flush=True)
    return {
        "L": L_trace[post_warmup], "F": F_trace[post_warmup], "alpha_p": alpha_trace[post_warmup],
        "rho": rho_trace[post_warmup], "sigma_alpha_sq": sigma_trace[post_warmup],
    }


def summarize(name, ours, reference):
    """Compare a scalar functional's mean/std/ESS between the two chains."""
    ours_ess = float(effective_sample_size(ours))
    reference_ess = float(effective_sample_size(reference))
    ours_mean, ours_std = float(jnp.mean(ours)), float(jnp.std(ours))
    reference_mean, reference_std = float(np.mean(reference)), float(np.std(reference))
    pooled_se = np.sqrt(ours_std**2 / max(ours_ess, 1.0) + reference_std**2 / max(reference_ess, 1.0))
    z = (ours_mean - reference_mean) / pooled_se if pooled_se > 0 else float("nan")
    return {
        "ours_mean": ours_mean, "ours_std": ours_std, "ours_ess": ours_ess,
        "reference_mean": reference_mean, "reference_std": reference_std, "reference_ess": reference_ess,
        "z": z, "status": "PASS" if abs(z) < 3.0 else "WARN",
    }


def main():
    """Draw a slide, run both chains, and report the cross-check."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    coordinates, spot_graph, pathway_graph, pathway_laplacian = build_graphs(args, rng)

    key = jax.random.PRNGKey(args.seed)
    prior_key, train_key, reference_key, our_key = jax.random.split(key, 4)
    prior = draw_representative_prior(prior_key, args, spot_graph, pathway_laplacian)

    print("warm-starting amortized heads...", flush=True)
    spatial, loading = warm_start_heads(train_key, args, spot_graph, coordinates, pathway_graph, pathway_laplacian)

    print("running NumPyro reference chain...", flush=True)
    reference_samples = run_reference_chain(reference_key, args, prior, spot_graph, pathway_laplacian)

    print("running our mixture-kernel + Gibbs chain...", flush=True)
    our_samples = run_our_chain(our_key, args, prior, spot_graph, pathway_graph, pathway_laplacian, spatial, loading)

    report = {
        "mean_L": summarize("mean_L", jnp.mean(our_samples["L"], axis=(1, 2)), np.mean(reference_samples["L"], axis=(1, 2))),
        "mean_F": summarize("mean_F", jnp.mean(our_samples["F"], axis=(1, 2)), np.mean(reference_samples["F"], axis=(1, 2))),
        "mean_alpha_p": summarize("mean_alpha_p", jnp.mean(our_samples["alpha_p"], axis=1), np.mean(reference_samples["alpha_p"], axis=1)),
        "rho": summarize("rho", our_samples["rho"], reference_samples["rho"]),
        "sigma_alpha_sq": summarize("sigma_alpha_sq", our_samples["sigma_alpha_sq"], reference_samples["sigma_alpha_sq"]),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
