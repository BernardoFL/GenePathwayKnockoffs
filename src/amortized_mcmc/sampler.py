"""Chain-state containers and exact proposal-to-MH adapters."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import blackjax

from .graph import Graph
from .models import AmortizedProposal
from .csp import CSPState


class ChainState(NamedTuple):
    """State carried by spatial/loading updates and auxiliary Gibbs steps.

    ``csp`` is optional for backwards-compatible L-only experiments but should
    be populated for the full model.
    """
    X: jax.Array
    L: jax.Array
    F: jax.Array
    alpha: jax.Array
    p_of_s: jax.Array
    graph: Graph
    csp: CSPState | None = None


def propose_L(key: jax.Array, X: jax.Array, L: jax.Array, graph: Graph, q_module) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample an L proposal and evaluate forward and reverse log densities."""
    proposal, log_q_forward = q_module.sample_and_log_q(key, X, L, graph)
    log_q_reverse = q_module.log_q(L, X, proposal, graph)
    return proposal, log_q_forward, log_q_reverse


def propose_F(key: jax.Array, X: jax.Array, F: jax.Array, q_module) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample an F proposal and evaluate both transition directions."""
    proposal, log_q_forward = q_module.sample_and_log_q(key, X, F)
    log_q_reverse = q_module.log_q(F, X, proposal)
    return proposal, log_q_forward, log_q_reverse


def mh_step(
    key: jax.Array,
    state: ChainState,
    q_module,
    pi_fn: Callable[[jax.Array, ChainState], jax.Array],
) -> tuple[ChainState, jax.Array, jax.Array]:
    """Perform one explicit exact MH update of ``L``."""
    proposal_key, accept_key = jax.random.split(key)
    proposed, log_q_forward, log_q_reverse = propose_L(
        proposal_key, state.X, state.L, state.graph, q_module
    )
    proposed_state = state._replace(L=proposed)
    log_ratio = (
        pi_fn(proposed, state) + log_q_reverse - pi_fn(state.L, state) - log_q_forward
    )
    log_acceptance = jnp.minimum(0.0, log_ratio)
    accepted = jnp.log(jax.random.uniform(accept_key)) < log_acceptance
    return jax.lax.cond(accepted, lambda _: (proposed_state, True, jnp.exp(log_acceptance)), lambda _: (state, False, jnp.exp(log_acceptance)), operand=None)


def blackjax_mh_step(
    key: jax.Array,
    state: ChainState,
    q_module,
    logdensity_fn: Callable[[jax.Array], jax.Array],
) -> tuple[ChainState, jax.Array, jax.Array]:
    """Run one asymmetric amortized proposal through BlackJAX RMH.

    ``logdensity_fn`` must be the true target for L. Proposal parameters only
    enter the transition generator and reverse-density callback.
    The callback must return the true log density for the current L state;
    proposal parameters are used only by the transition generator and reverse
    proposal-density callback.
    """
    def proposal_generator(proposal_key, current_L):
        """Generate a candidate L using the conditional flow."""
        proposal, _ = q_module.sample_and_log_q(
            proposal_key, state.X, current_L, state.graph
        )
        return proposal

    def proposal_logdensity(proposed_blackjax_state, previous_blackjax_state):
        """Evaluate q(previous | proposed), as required by MH reversal."""
        return q_module.log_q(
            previous_blackjax_state.position,
            state.X,
            proposed_blackjax_state.position,
            state.graph,
        )

    algorithm = blackjax.rmh(logdensity_fn, proposal_generator, proposal_logdensity)
    initial = algorithm.init(state.L)
    next_blackjax_state, info = algorithm.step(key, initial)
    next_state = state._replace(L=next_blackjax_state.position)
    return next_state, info.is_accepted, info.acceptance_rate
