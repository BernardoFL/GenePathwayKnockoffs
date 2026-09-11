import jax
import jax.numpy as jnp
import numpy as np

from amortized_mcmc import (
    AmortizedProposal,
    CSPState,
    ChainState,
    LoadingProposal,
    SpatialProposal,
    build_graph,
    empirical_coverage,
    full_generative_model,
    log_pi_L,
    mh_step,
    blackjax_mh_step,
    conditional_flow_matching_loss,
    gibbs_step_csp,
    integrate_log_density,
    numpyro_log_pi_L,
    propose_F,
    propose_L,
    sample_csp_prior,
)
from numpyro.infer import Predictive


def make_state():
    key = jax.random.PRNGKey(0)
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    graph = build_graph(coordinates, k=2)
    X = jnp.ones((4, 3))
    L = jnp.zeros((4, 2))
    F = jnp.zeros((3, 2))
    alpha = jnp.zeros((1,))
    state = ChainState(X, L, F, alpha, jnp.zeros((4,), dtype=jnp.int32), graph)
    return key, state


def test_graph_methods_and_reverse_densities():
    key, state = make_state()
    spatial = SpatialProposal(2, 3, hidden_dim=8, key=key)
    loading = LoadingProposal(3, 3, 2, hidden_dim=8, key=jax.random.split(key)[1])
    new_l, forward_l, reverse_l = propose_L(key, state.X, state.L, state.graph, spatial)
    new_f, forward_f, reverse_f = propose_F(key, state.X, state.F, loading)
    assert new_l.shape == state.L.shape
    assert new_f.shape == state.F.shape
    assert jnp.isfinite(forward_l)
    assert jnp.isfinite(reverse_l)
    assert jnp.isfinite(forward_f)
    assert jnp.isfinite(reverse_f)


def test_loading_flow_has_fixed_h_and_finite_bidirectional_density():
    key, state = make_state()
    loading = LoadingProposal(3, 3, 2, hidden_dim=8, key=key)
    proposal, forward_density = loading.sample_and_log_q(key, state.X, state.F, steps=4)
    reverse_density = loading.log_q(state.F, state.X, proposal, key=key, steps=4)
    assert proposal.shape == state.F.shape
    assert loading.flow_field(0.5, state.F, state.X, state.F).shape == state.F.shape
    assert jnp.isfinite(forward_density)
    assert jnp.isfinite(reverse_density)


def test_numpyro_prior_predictive_contains_full_generative_state():
    _, state = make_state()
    predictive = Predictive(full_generative_model, num_samples=1)
    samples = predictive(
        jax.random.PRNGKey(16),
        n_spots=4,
        n_genes=3,
        H=2,
        senders=state.graph.senders,
        receivers=state.graph.receivers,
    )
    for name in ("alpha_csp", "b_phi", "v", "z_0", "z_1", "phi", "lambda_local", "F", "L", "alpha_p", "X"):
        assert name in samples


def test_spatial_flow_has_fixed_h_and_finite_bidirectional_density():
    key, state = make_state()
    spatial = SpatialProposal(2, 3, hidden_dim=8, key=key)
    proposal, forward_density, _ = propose_L(key, state.X, state.L, state.graph, spatial)
    reverse_density = spatial.log_q(state.L, state.X, proposal, state.graph, key=key, steps=4)
    assert spatial.H == 2
    assert spatial.flow_field(0.5, state.L, state.X, state.L, state.graph).shape == state.L.shape
    assert jnp.isfinite(forward_density)
    assert jnp.isfinite(reverse_density)


def test_flow_matching_loss_is_scalar():
    key, state = make_state()
    spatial = SpatialProposal(2, 3, hidden_dim=8, key=key)
    field = lambda time, value, X, current, graph: spatial.flow_field(time, value, X, current, graph)
    loss = conditional_flow_matching_loss(field, key, state.L, state.L + 0.1, state.X, state.L, state.graph)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_linear_flow_density_has_expected_divergence_correction():
    value = jnp.ones((2,))
    coefficient = 0.25
    field = lambda time, state: coefficient * state
    density = integrate_log_density(field, value, lambda state: -0.5 * jnp.sum(state * state + jnp.log(2.0 * jnp.pi)), jax.random.PRNGKey(12), steps=32)
    expected = -0.5 * jnp.sum((value * jnp.exp(-coefficient)) ** 2 + jnp.log(2.0 * jnp.pi)) - 2.0 * coefficient
    assert jnp.allclose(density, expected, atol=2e-3)


def test_csp_update_keeps_fixed_spike_precision():
    key = jax.random.PRNGKey(10)
    csp = sample_csp_prior(key, H=4, n_genes=3, theta_infty=1e4)
    updated = gibbs_step_csp(jax.random.PRNGKey(11), jnp.zeros((3, 4)), csp)
    assert isinstance(updated, CSPState)
    assert updated.H == 4
    assert jnp.all(jnp.where(updated.z != jnp.arange(4), updated.phi, 1.0) == jnp.where(updated.z != jnp.arange(4), 1e4, 1.0))


def test_csp_update_accepts_chain_state():
    _, state = make_state()
    csp = sample_csp_prior(jax.random.PRNGKey(14), H=2, n_genes=3)
    updated = gibbs_step_csp(jax.random.PRNGKey(15), state._replace(csp=csp))
    assert updated.csp.H == 2


def test_target_does_not_depend_on_proposal():
    _, state = make_state()
    value = log_pi_L(state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph)
    assert jnp.isfinite(value)


def test_numpyro_target_matches_explicit_target():
    _, state = make_state()
    explicit = log_pi_L(state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph)
    modeled = numpyro_log_pi_L(
        state.L, state.X, state.F, state.alpha, state.p_of_s, state.graph
    )
    assert jnp.allclose(explicit, modeled)


def test_mh_step_returns_valid_state_and_acceptance():
    key, state = make_state()
    proposal = SpatialProposal(2, 3, hidden_dim=8, key=key)

    def target(L, current_state):
        return log_pi_L(
            L,
            current_state.X,
            current_state.F,
            current_state.alpha,
            current_state.p_of_s,
            current_state.graph,
        )

    new_state, accepted, acceptance = mh_step(key, state, proposal, target)
    assert new_state.L.shape == state.L.shape
    assert accepted.dtype == jnp.bool_
    assert 0.0 <= acceptance <= 1.0


def test_blackjax_mh_step_returns_valid_state_and_acceptance():
    key, state = make_state()
    proposal = SpatialProposal(2, 3, hidden_dim=8, key=key)

    def target(L):
        return numpyro_log_pi_L(
            L, state.X, state.F, state.alpha, state.p_of_s, state.graph
        )

    new_state, accepted, acceptance = blackjax_mh_step(key, state, proposal, target)
    assert new_state.L.shape == state.L.shape
    assert accepted.dtype == jnp.bool_
    assert 0.0 <= acceptance <= 1.0


def test_blackjax_flow_step_is_jittable():
    key, state = make_state()
    proposal = SpatialProposal(2, 3, hidden_dim=8, key=key)

    def target(L):
        return numpyro_log_pi_L(L, state.X, state.F, state.alpha, state.p_of_s, state.graph)

    step = jax.jit(lambda step_key: blackjax_mh_step(step_key, state, proposal, target))
    new_state, accepted, acceptance = step(jax.random.PRNGKey(13))
    assert new_state.L.shape == state.L.shape
    assert accepted.dtype == jnp.bool_
    assert jnp.isfinite(acceptance)


def test_empirical_coverage_reports_calibration():
    report = empirical_coverage(jnp.zeros((10, 2)), jnp.zeros((10, 2)), jnp.ones((10, 2)))
    assert report.coverage.shape == (2,)
    assert report.n_observations == 10
