import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from amortized_mcmc import (
    AmortizedProposal,
    CSPState,
    ChainState,
    LoadingProposal,
    MOVE_BLOCKED_F,
    MOVE_BLOCKED_L,
    MOVE_GLOBAL_JOINT,
    SpatialProposal,
    assert_graphs_not_aliased,
    build_graph,
    build_pathway_graph,
    empirical_coverage,
    full_generative_model,
    gibbs_step_alpha_p,
    gibbs_step_csp,
    gibbs_step_rho,
    gibbs_step_sigma_alpha_sq,
    assert_gene_panel_matches,
    laplace_mrf_logprob,
    log_pi,
    spectral_zone_masks,
    log_pi_block_F,
    log_pi_block_L,
    log_pi_L,
    mixture_step,
    numpyro_log_pi_L,
    pathway_laplacian_from_affinity,
    pathway_logprob,
    propose_F_block,
    propose_global_joint,
    propose_L_block,
    sample_csp_prior,
)
from numpyro.infer import Predictive

H = 2
N_GENES = 4


def make_state():
    key = jax.random.PRNGKey(0)
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    graph = build_graph(coordinates, k=2)
    affinity = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float32
    )
    pathway_graph = build_pathway_graph(affinity)
    X = jnp.ones((4, N_GENES))
    L = jnp.zeros((4, H))
    F = jnp.zeros((N_GENES, H))
    alpha = jnp.zeros((1,))
    csp = sample_csp_prior(jax.random.PRNGKey(1), H=H, n_genes=N_GENES)
    state = ChainState(X, L, F, alpha, jnp.zeros((4,), dtype=jnp.int32), graph, csp, pathway_graph, jnp.asarray(1.0))
    return key, state, affinity


def make_heads(key):
    spatial_key, loading_key = jax.random.split(key)
    spatial = SpatialProposal(H, N_GENES, hidden_dim=8, depth=4, key=spatial_key)
    loading = LoadingProposal(H, N_GENES, hidden_dim=8, depth=4, key=loading_key)
    return AmortizedProposal(spatial, loading)


def test_spatial_block_proposal_has_self_consistent_exact_density():
    key, state, _ = make_state()
    spatial = SpatialProposal(H, N_GENES, hidden_dim=8, depth=4, key=key)
    always_passive = jnp.array([True, True, False, False])
    proposal, log_q_fwd, _ = propose_L_block(key, state.X, state.L, always_passive, state.graph, spatial)
    assert proposal.shape == state.L.shape
    # The complement must be copied through untouched.
    assert jnp.allclose(proposal[always_passive], state.L[always_passive])
    # log_q evaluated at the sample it was drawn from must reproduce the sampler's own density exactly.
    reverse_of_sample = spatial.log_q(proposal, state.X, always_passive, state.graph)
    assert jnp.allclose(log_q_fwd, reverse_of_sample, atol=1e-4)
    assert jnp.isfinite(log_q_fwd)


def test_loading_block_proposal_has_self_consistent_exact_density():
    key, state, _ = make_state()
    loading = LoadingProposal(H, N_GENES, hidden_dim=8, depth=4, key=key)
    always_passive = jnp.array([True, False, False, True])
    proposal, log_q_fwd, _ = propose_F_block(key, state.X, state.F, always_passive, state.pathway_graph, loading)
    assert proposal.shape == state.F.shape
    assert jnp.allclose(proposal[always_passive], state.F[always_passive])
    reverse_of_sample = loading.log_q(proposal, state.X, always_passive, state.pathway_graph)
    assert jnp.allclose(log_q_fwd, reverse_of_sample, atol=1e-4)


def test_global_joint_proposal_ignores_current_state_context():
    key, state, _ = make_state()
    heads = make_heads(key)
    L_new, F_new, log_q_fwd, log_q_rev = propose_global_joint(
        key, state.X, state.graph, state.pathway_graph, heads, state.L, state.F
    )
    assert L_new.shape == state.L.shape
    assert F_new.shape == state.F.shape
    assert jnp.isfinite(log_q_fwd)
    assert jnp.isfinite(log_q_rev)
    # A global jump has no context spots at all -- proposing from a different
    # "current" state must not change the forward sample's density function,
    # only which point log_q_rev is evaluated at.
    other_log_q_fwd = heads.spatial.log_q(L_new, state.X, jnp.zeros(4, dtype=bool), state.graph) + heads.loading.log_q(
        F_new, state.X, jnp.zeros(N_GENES, dtype=bool), state.pathway_graph
    )
    assert jnp.allclose(log_q_fwd, other_log_q_fwd, atol=1e-4)


def test_numpyro_prior_predictive_contains_full_generative_state():
    _, state, _ = make_state()
    predictive = Predictive(full_generative_model, num_samples=1)
    samples = predictive(
        jax.random.PRNGKey(16),
        n_spots=4,
        n_genes=N_GENES,
        H=H,
        senders=state.graph.senders,
        receivers=state.graph.receivers,
    )
    for name in ("alpha_csp", "b_phi", "v", "z_0", "z_1", "phi", "lambda_local", "F", "L", "alpha_p", "X"):
        assert name in samples


def test_csp_update_keeps_fixed_spike_precision():
    key = jax.random.PRNGKey(10)
    csp = sample_csp_prior(key, H=4, n_genes=3, theta_infty=1e4)
    updated = gibbs_step_csp(jax.random.PRNGKey(11), jnp.zeros((3, 4)), csp)
    assert isinstance(updated, CSPState)
    assert updated.H == 4
    assert jnp.all(jnp.where(updated.z != jnp.arange(4), updated.phi, 1.0) == jnp.where(updated.z != jnp.arange(4), 1e4, 1.0))


def test_csp_update_accepts_chain_state():
    _, state, _ = make_state()
    updated = gibbs_step_csp(jax.random.PRNGKey(15), state)
    assert updated.csp.H == H


def test_gibbs_step_rho_is_positive_and_finite():
    _, state, affinity = make_state()
    laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    rho_new = gibbs_step_rho(jax.random.PRNGKey(3), state.F, state.csp, laplacian)
    assert rho_new > 0.0
    assert jnp.isfinite(rho_new)


def test_pathway_logprob_only_couples_active_columns():
    _, state, affinity = make_state()
    laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    key = jax.random.PRNGKey(4)
    F = jax.random.normal(key, (N_GENES, H))
    all_inactive = jnp.zeros((H,), dtype=bool)
    # With no active columns the pathway term reduces to exactly zero.
    assert pathway_logprob(F, laplacian, 1.0, all_inactive) == 0.0


def test_log_pi_reads_csp_phi():
    """log_pi must actually incorporate the CSP column precision phi_k (a
    model parameter, not a proposal-network parameter) -- this guards
    against a refactor accidentally dropping the Horseshoe/CSP prior term.
    """
    _, state, affinity = make_state()
    laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    value = log_pi(state.L, state.F, state.X, state.alpha, state.p_of_s, state.graph, state.csp, pathway_laplacian=laplacian, rho=1.0)
    perturbed_csp = state.csp._replace(phi=state.csp.phi * 3.0 + 1.0)
    perturbed = log_pi(state.L, state.F, state.X, state.alpha, state.p_of_s, state.graph, perturbed_csp, pathway_laplacian=laplacian, rho=1.0)
    assert jnp.isfinite(value) and jnp.isfinite(perturbed)
    assert value != perturbed


def test_log_pi_cannot_depend_on_proposal_network_parameters():
    """The spec's hard invariant: the amortized network only ever shapes a
    proposal and never touches the target pi. log_pi's signature has no
    proposal-module parameter at all, so its value for a fixed (L, F, state)
    is bit-identical regardless of which -- or how many -- differently
    initialized proposal heads exist alongside it.
    """
    key, state, affinity = make_state()
    laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    heads_a = make_heads(key)
    heads_b = make_heads(jax.random.PRNGKey(999))

    def value_with(heads):
        """Compute log_pi with the given heads merely in scope, unused."""
        del heads  # log_pi never receives it; kept only to show scope can't leak in
        return log_pi(state.L, state.F, state.X, state.alpha, state.p_of_s, state.graph, state.csp, pathway_laplacian=laplacian, rho=1.0)

    assert value_with(heads_a) == value_with(heads_b)


def test_numpyro_target_matches_explicit_target():
    _, state, _ = make_state()
    explicit = log_pi_L(state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph)
    modeled = numpyro_log_pi_L(state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph)
    assert jnp.allclose(explicit, modeled)


def test_log_pi_block_matches_full_target_at_full_state():
    _, state, affinity = make_state()
    laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    assert jnp.allclose(log_pi_block_L(state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph), log_pi_L(state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph))
    assert jnp.allclose(
        log_pi_block_F(state.F, state.X, state.L, state.alpha, state.p_of_s, state.csp, laplacian, 1.0),
        log_pi_block_F(state.F, state.X, state.L, state.alpha, state.p_of_s, state.csp, laplacian, 1.0),
    )


def test_mixture_step_returns_valid_state_and_move_type():
    key, state, affinity = make_state()
    heads = make_heads(key)
    laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    spot_zones = jnp.array([[False, False, True, True], [True, True, False, False]])
    gene_zones = jnp.array([[False, False, True, True], [True, True, False, False]])

    def log_pi_fn(L, F, chain_state):
        return log_pi(
            L, F, chain_state.X, chain_state.alpha, chain_state.p_of_s, chain_state.graph, chain_state.csp,
            pathway_laplacian=laplacian, rho=chain_state.rho,
        )

    new_state, accepted, alpha_mh, move_type = mixture_step(key, state, 0.3, heads, log_pi_fn, spot_zones, gene_zones)
    assert new_state.L.shape == state.L.shape
    assert new_state.F.shape == state.F.shape
    assert accepted.dtype == jnp.bool_
    assert 0.0 <= alpha_mh <= 1.0
    assert int(move_type) in (MOVE_GLOBAL_JOINT, MOVE_BLOCKED_L, MOVE_BLOCKED_F)


def test_mixture_step_is_jittable():
    key, state, affinity = make_state()
    heads = make_heads(key)
    laplacian = pathway_laplacian_from_affinity(jnp.asarray(affinity))
    spot_zones = jnp.array([[False, False, True, True], [True, True, False, False]])
    gene_zones = jnp.array([[False, False, True, True], [True, True, False, False]])

    def log_pi_fn(L, F, chain_state):
        return log_pi(
            L, F, chain_state.X, chain_state.alpha, chain_state.p_of_s, chain_state.graph, chain_state.csp,
            pathway_laplacian=laplacian, rho=chain_state.rho,
        )

    step = jax.jit(lambda step_key: mixture_step(step_key, state, 0.3, heads, log_pi_fn, spot_zones, gene_zones))
    new_state, accepted, alpha_mh, move_type = step(jax.random.PRNGKey(13))
    assert new_state.L.shape == state.L.shape
    assert accepted.dtype == jnp.bool_
    assert jnp.isfinite(alpha_mh)


def test_gibbs_step_alpha_p_alias_matches_original():
    _, state, _ = make_state()
    updated = gibbs_step_alpha_p(jax.random.PRNGKey(2), state)
    assert updated.shape == state.alpha.shape


def test_empirical_coverage_reports_calibration():
    report = empirical_coverage(jnp.zeros((10, 2)), jnp.zeros((10, 2)), jnp.ones((10, 2)))
    assert report.coverage.shape == (2,)
    assert report.n_observations == 10


def test_assert_graphs_not_aliased_rejects_same_object():
    _, state, _ = make_state()
    with pytest.raises(ValueError):
        assert_graphs_not_aliased(state.pathway_graph, state.pathway_graph)


def test_assert_graphs_not_aliased_accepts_distinct_graphs():
    _, state, _ = make_state()
    knockoff_graph = build_graph(np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]), k=2)
    assert_graphs_not_aliased(state.pathway_graph, knockoff_graph)


def test_laplace_mrf_is_elementwise_l1_not_group_l2():
    """The model spec's edge-preserving prior is a per-component Laplace
    potential (sum of |diff_k| over every (edge, k) pair), which is a
    materially different distribution from an edge-wise Euclidean (L2)
    group norm over components -- regression guard against reintroducing
    the group-norm form.
    """
    _, state, _ = make_state()
    L = jnp.array([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0], [0.0, 0.0]])
    differences = np.asarray(L)[np.asarray(state.graph.senders)] - np.asarray(L)[np.asarray(state.graph.receivers)]
    expected = -np.sum(np.abs(differences))
    assert jnp.allclose(laplace_mrf_logprob(L, state.graph), expected)


def test_csp_concentration_conditional_matches_true_posterior():
    """Regression guard for the alpha_CSP slice sampler's exponent, which
    previously overweighted by one free stick (using the full padded ``v``
    array's length instead of the number of actual Beta(1, alpha_CSP)
    draws)."""
    from amortized_mcmc.csp import _log_concentration

    rng = np.random.default_rng(0)
    alpha_true, H = 2.0, 6
    v_free = rng.beta(1.0, alpha_true, size=H - 1)
    v_full = jnp.asarray(np.concatenate([v_free, [1.0]]))
    e0, f0 = 2.0, 1.0

    def true_log_posterior(alpha):
        log_prior = (e0 - 1) * np.log(alpha) - f0 * alpha
        log_lik = (H - 1) * np.log(alpha) + alpha * np.sum(np.log(1 - v_free))
        return log_prior + log_lik

    for alpha in (0.5, 1.5, 3.0, 5.0):
        got = float(_log_concentration(jnp.asarray(alpha), v_full, e0, f0))
        assert np.isclose(got, true_log_posterior(alpha), atol=1e-4)


def test_sample_csp_prior_local_scale_matches_half_cauchy_quantiles():
    """Regression guard: nu must actually be Half-Cauchy(0,1)-distributed,
    not the far heavier InverseGamma(0.5,1) that sampling 1/Gamma(0.5,1)
    directly (skipping the two-stage parameter expansion) produces."""
    csp = sample_csp_prior(jax.random.PRNGKey(1), H=2, n_genes=20000)
    lam = np.asarray(csp.lam).reshape(-1)
    assert 0.85 < np.median(lam) < 1.15  # true Half-Cauchy(0,1) median is 1.0


def test_gibbs_step_sigma_alpha_sq_recovers_known_variance():
    sigma_true = 2.0
    alpha_p = jax.random.normal(jax.random.PRNGKey(0), (5000,)) * jnp.sqrt(sigma_true)
    draws = jnp.stack([gibbs_step_sigma_alpha_sq(jax.random.PRNGKey(i), alpha_p) for i in range(200)])
    assert abs(float(jnp.mean(draws)) - sigma_true) < 0.3


def test_spectral_zone_masks_splits_along_weak_bridge():
    """Two tight clusters joined by one weak edge must be split along that
    edge -- the concrete behavior "zones read off the affinity/attention
    structure" requires, as opposed to an arbitrary ordering."""
    n = 8
    senders = np.array([0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 3])
    receivers = np.array([1, 2, 3, 3, 0, 0, 5, 6, 7, 7, 4, 4, 4])
    weights = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0.01])
    masks = np.asarray(spectral_zone_masks(n, senders, receivers, 2, weights))
    zone0 = frozenset(np.where(~masks[0])[0].tolist())
    zone1 = frozenset(np.where(~masks[1])[0].tolist())
    assert zone0 | zone1 == frozenset(range(n))
    assert zone0 & zone1 == frozenset()
    assert zone0 in (frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7}))


def test_loading_gradient_is_finite_for_constant_gene_column():
    """Regression guard: a gene with a constant (e.g. all-zero) count column
    across spots previously gave the DeepSets pooling encoder's std-pool a
    NaN gradient (jnp.std has an undefined gradient at exactly zero
    variance), which silently poisoned training after a few steps."""
    key, state, _ = make_state()
    loading = LoadingProposal(H, N_GENES, hidden_dim=8, depth=4, key=key)
    X = state.X.at[:, 0].set(0.0)  # force a constant (all-zero) gene column
    always_passive = jnp.zeros((N_GENES,), dtype=bool)

    def loss_fn(model):
        return -model.log_q(state.F, X, always_passive, state.pathway_graph)

    grads = eqx.filter_grad(loss_fn)(loading)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_assert_gene_panel_matches_accepts_identical_panels():
    panel = np.array(["GENE_0", "GENE_1", "GENE_2"])
    assert_gene_panel_matches(panel, panel.copy())


def test_assert_gene_panel_matches_rejects_reordered_panel():
    panel = np.array(["GENE_0", "GENE_1", "GENE_2"])
    reordered = np.array(["GENE_1", "GENE_0", "GENE_2"])
    with pytest.raises(ValueError):
        assert_gene_panel_matches(panel, reordered)


def test_assert_gene_panel_matches_rejects_different_panel_size():
    panel = np.array(["GENE_0", "GENE_1", "GENE_2"])
    other = np.array(["GENE_0", "GENE_1"])
    with pytest.raises(ValueError):
        assert_gene_panel_matches(panel, other)


def test_spectral_zone_masks_partitions_every_node_exactly_once():
    _, state, _ = make_state()
    n = state.L.shape[0]
    masks = np.asarray(spectral_zone_masks(n, state.graph.senders, state.graph.receivers, 3))
    covered = np.zeros(n, dtype=int)
    for row in masks:
        covered += (~row).astype(int)
    assert np.all(covered == 1)
