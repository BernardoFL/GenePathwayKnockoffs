# Amortized Neural MCMC

JAX + Equinox coupling-flow proposal modules for exact Metropolis-Hastings updates of spatial factors `L` and gene loadings `F`. The networks only ever shape a proposal; `log_pi` remains independent of proposal parameters, and both directions of the proposal density (`q_phi(new | ...)` and `q_phi(current | ...)`) come from inverting the same deterministic coupling flow -- never an ODE, never a stochastic trace estimator.

## Architecture

The project has two workflows. The offline workflow draws prior-simulation
training data and fits both amortizer heads by maximum-likelihood flow
density estimation. The inference workflow loads the trained heads, proposes
zone-blocked or global-joint independence draws, and applies the exact
target-density correction.

```mermaid
flowchart TD
	subgraph DATA[Offline prior-simulation data generation]
		C[Coordinates] --> G[graph.py\nk-NN or Delaunay]
		G --> GM[Graphs\nnode and edge structure]
		P[generative_model.py\nNumPyro full prior + likelihood] --> PR[Predictive\nprior draw only]
		PR --> S0[(L, F, alpha_p, CSP, rho, X)]
		S0 --> N[simulate_prior.py\nflattened jraph NPZ shard]
		GM --> N
	end

	subgraph TRAIN[Offline amortizer training]
		N --> JL[jraph GraphsTuple\nbatch/unbatch loader]
		JL --> L1[SpatialProposal\nGAT-conditioned graph coupling flow]
		JL --> L2[LoadingProposal\nGAT-conditioned graph coupling flow, Student-t base]
		L1 --> MLE[train_amortizer.py\n-log q_phi maximum likelihood]
		L2 --> MLE
		MLE --> OPT[Optax optimization]
		OPT --> CKPT[(Equinox checkpoints)]
	end

	subgraph INFER[Exact posterior inference]
		CKPT --> QL[Head 1: L graph coupling flow]
		CKPT --> QF[Head 2: F graph coupling flow]
		QL --> INV[flow.py\nGraphCouplingLayer: one-pass sample + exact inverse log-det]
		QF --> INV
		INV --> PROP[propose_L_block / propose_F_block / propose_global_joint]
		PROP --> MIX[mixture_step\n(1-beta) zone-blocked, beta global joint]
		TARGET[target.py / numpyro_model.py\ntrue likelihood + priors, independent of phi]
		TARGET --> MIX
		MIX --> STATE[ChainState]
		STATE --> A[alpha.py\nalpha_p slice update]
		STATE --> CSP[csp.py\nCSP Gibbs update]
		STATE --> RHO[pathway.py\nrho Gibbs update]
		A --> STATE
		CSP --> STATE
		RHO --> STATE
	end

	style TARGET fill:#f9d5d3,stroke:#a33
	style MIX fill:#d7ead9,stroke:#286b35
	style CKPT fill:#d9e8fb,stroke:#3569a8
```

### Module responsibilities

| Module | Responsibility |
| --- | --- |
| `graph.py` | Builds spatial spot graphs and pathway-annotation graphs; asserts they never alias the (separate) knockoff test graph. |
| `generative_model.py` | Defines the NumPyro prior, CSP hierarchy, MRF prior, pathway-coupled `F`, patient effects, and Poisson likelihood. |
| `simulate_prior.py` | Draws prior-only training examples with `Predictive` -- no reference MCMC, no transition pairs. |
| `models.py` | Defines the two GAT-conditioned graph coupling flow heads (spatial, pathway). |
| `flow.py` | Exact, deterministic coupling-flow primitives: one-pass sample + inverse log-det, no ODE or trace estimator. |
| `train_amortizer.py` | Loads a prior-simulation npz shard and fits both heads by maximum-likelihood flow density estimation, with random context masking so one network serves both the global-jump and zone-blocked conditional modes. |
| `sampler.py` | The zone-blocked / global-joint mixture kernel and its exact Metropolis-Hastings acceptance. |
| `target.py` / `numpyro_model.py` | Evaluate the true target, independently of amortizer parameters. |
| `alpha.py` | Updates patient effects outside the neural proposal. |
| `csp.py` | Updates CSP allocation and shrinkage variables outside the neural proposal. |
| `pathway.py` | Updates the pathway-coupling strength `rho` outside the neural proposal. |
| `panel.py` | Asserts a simulator's gene panel matches a real dataset's before training data crosses that boundary. |
| `zoning.py` | Reads zone-blocked update partitions off a graph's (optionally attention-weighted) structure via recursive spectral bisection. |
| `diagnostics.py` / `mixing.py` | Coverage, boundary acceptance, ESS, and MH-versus-Gibbs diagnostics. |
| `calibrate.py` | Calibration harness: empirical coverage and SBC-rank diagnostics for both heads, stratified by boundary proximity for Head 1. |
| `reference_check.py` | Cross-checks the exact sampler's posterior summaries against a NumPyro `DiscreteHMCGibbs(NUTS(...))` reference chain on a small slide. |

## Install

```bash
conda env create -f environment.yml
conda activate genepathway
```

## Scope

- `build_graph(..., method='knn'|'delaunay')` constructs the non-grid spot graph; `build_pathway_graph` builds the gene-gene pathway graph.
- `SpatialProposal` (Head 1) is a GAT-conditioned graph coupling flow over the spot graph, agnostic to spot count.
- `LoadingProposal` (Head 2) is a GAT-conditioned graph coupling flow over the pathway graph, with a Student-t base by default so its tails dominate the target's.
- `propose_L_block` / `propose_F_block` sample a zone-blocked independence proposal conditioned only on the block's complement, and score both directions exactly.
- `propose_global_joint` samples a context-free joint `(L, F)` jump.
- `mixture_step` composes these into the exact `(1-beta)`-blocked / `beta`-global kernel.
- `sample_csp_prior` and `gibbs_step_csp` handle the fixed-truncation CSP outside the network; `gibbs_step_rho` updates the pathway-coupling strength outside the network.
- `spatial_model` and `numpyro_log_pi_L` expose the L target through NumPyro.
- `gibbs_step_alpha_p` updates `alpha_p` without using either network. The Poisson-log-Gaussian conditional is not conjugate, so this is an exact slice update rather than a closed-form Gaussian draw.

## Amortizer Training

The amortizer is trained separately from posterior sampling. Generate a
prior-simulation shard (every example is one `Predictive` prior draw --
no reference chain, no transition pairs):

```bash
python scripts/simulate_prior.py \
	--output shard.npz --n-examples 10000 --H 8 --n-genes 2000
```

Then fit both heads by maximum-likelihood flow density estimation:

```bash
conda activate genepathway
python scripts/train_amortizer.py \
	--data shard.npz \
	--output results/amortizer \
	--epochs 10000
```

Each step draws a random per-example context mask: with probability
`--empty-context-prob` it is empty (training the global-jump mode
`q_phi(. | X)`); otherwise a random node subset is held out as fixed context
(training the zone-blocked conditional mode `q_phi(free | X, context)`).
One network serves both proposal modes.

For Slurm:

```bash
DATA=/path/shard.npz OUTPUT=/path/results/amortizer sbatch scripts/submit_amortizer_slurm.sh
```

This job only optimizes the two coupling-flow heads. It does not run MH,
Gibbs, alpha, CSP, rho, or posterior sampling.

## Posterior Sampling

```bash
python scripts/run_hpc.py \
	--data slide.npz --output results/chain \
	--spatial-checkpoint results/amortizer/spatial_proposal_0010000.eqx \
	--loading-checkpoint results/amortizer/loading_proposal_0010000.eqx \
	--iterations 10000 --warmup 2000
```

`slide.npz` must contain `coordinates`, `X`, and `A_path` (the pathway
affinity matrix used both to build Head 2's graph and the pathway prior's
Laplacian). Architecture flags (`--hidden-dim`, `--depth`, `--gat-depth`,
`--spatial-base`, `--loading-base`, `--degrees-of-freedom`) must match the
values used at training time, since loading a checkpoint's serialized leaves
requires an identical skeleton model. Zone-blocked moves partition spots and
genes via `spectral_zone_masks`, weighted by each loaded head's own GAT
attention over its graph.

## Calibration and Reference-Chain Validation

Two scripts implement the §4/§10 validation deliverables and are meant to
run against a held-out shard and a real trained checkpoint, not just the
training shard:

```bash
python scripts/calibrate.py \
	--data held_out_shard.npz \
	--spatial-checkpoint results/amortizer/spatial_proposal_0010000.eqx \
	--loading-checkpoint results/amortizer/loading_proposal_0010000.eqx \
	--n-examples 200 --n-samples 128
```

Reports empirical coverage of q_phi's own credible intervals against the
shard's true prior draws, plus a simulation-based-calibration (SBC) rank
diagnostic (the rank of the true point's flow log-density among fresh
samples' log-densities -- concentration near the tails signals the
proposal's variance is too small, the failure mode that breaks the
independence sampler's geometric ergodicity), for both the global-jump and
a random zone-blocked context mode, and for Head 1, stratified by proximity
to a morphological boundary.

```bash
python scripts/reference_check.py --n-spots 12 --n-genes 6 --H 3
```

Cross-checks the exact `mixture_step` + Gibbs sampler's posterior summaries
against a NumPyro `DiscreteHMCGibbs(NUTS(full_generative_model))` reference
chain on one small synthetic slide (quickly warm-starting fresh heads
in-script by default; pass `--spatial-checkpoint`/`--loading-checkpoint` to
use real trained ones instead). This is a distributional smoke test, not a
proof of exact equivalence -- expect noisier agreement the less the
amortizer has actually been trained.
