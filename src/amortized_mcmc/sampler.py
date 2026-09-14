"""Chain-state container and the exact mixture-kernel MH transition.

Every proposal here is an amortized *independence* draw: it never conditions
on the value it is about to replace, only on ``X`` and (for a zone-blocked
move) the fixed complement outside the block. The corresponding
Metropolis-Hastings ratio is exact and deterministic -- ``log_pi`` never
touches proposal parameters, and both ``log_q`` directions come from
inverting the same coupling flow, never from an ODE or a stochastic trace
estimator. Gibbs updates for ``alpha_p``, the CSP allocation, and ``rho``
live in their own modules and are never touched by the network; they are
composed with this kernel by the caller, not by ``mixture_step`` itself.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .graph import Graph
from .models import AmortizedProposal, LoadingProposal, SpatialProposal
from .csp import CSPState


class ChainState(NamedTuple):
    """State carried by the (L, F) mixture kernel and the Gibbs blocks.

    ``csp``, ``pathway_graph``, and ``rho`` are optional so small
    L-only/F-only experiments can omit them, but all three must be populated
    to run the full model (``pathway_graph`` and ``rho`` are required by any
    move that touches ``F`` under the pathway prior).
    """
    X: jax.Array
    L: jax.Array
    F: jax.Array
    alpha: jax.Array
    p_of_s: jax.Array
    graph: Graph
    csp: CSPState | None = None
    pathway_graph: Graph | None = None
    rho: jax.Array | None = None


def propose_L_block(
    key: jax.Array,
    X: jax.Array,
    L_current: jax.Array,
    mask: jax.Array,
    spot_graph: Graph,
    spatial: SpatialProposal,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propose a fresh ``L`` for the spots outside ``mask``.

    ``mask`` is ``True`` at spots kept as fixed context (the block's
    complement); ``L_current`` supplies their values. Entries of
    ``L_current`` at ``~mask`` spots (the block) are read back only to score
    the reverse density -- they are never used as conditioning context for
    the forward draw. Returns the proposal (complement copied through,
    block freshly sampled) and the forward/reverse log-densities of the
    independence proposal restricted to the block.
    """
    proposal, log_q_forward = spatial.sample_and_log_q(key, X, mask, L_current, spot_graph)
    log_q_reverse = spatial.log_q(L_current, X, mask, spot_graph)
    return proposal, log_q_forward, log_q_reverse


def propose_F_block(
    key: jax.Array,
    X: jax.Array,
    F_current: jax.Array,
    mask: jax.Array,
    pathway_graph: Graph,
    loading: LoadingProposal,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propose a fresh ``F`` for the genes outside ``mask``; see :func:`propose_L_block`."""
    proposal, log_q_forward = loading.sample_and_log_q(key, X, mask, F_current, pathway_graph)
    log_q_reverse = loading.log_q(F_current, X, mask, pathway_graph)
    return proposal, log_q_forward, log_q_reverse


def propose_global_joint(
    key: jax.Array,
    X: jax.Array,
    spot_graph: Graph,
    pathway_graph: Graph,
    q_modules: AmortizedProposal,
    L_current: jax.Array,
    F_current: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Draw ``(L', F')`` independently of the current state and score both directions.

    No spot or gene is held as context: every node is free. ``L_current``
    and ``F_current`` are read back only to score the reverse density
    ``q_phi(L, F | X)`` at the current point, never as proposal context.
    """
    l_key, f_key = jax.random.split(key)
    no_context_L = jnp.zeros((L_current.shape[0],), dtype=bool)
    no_context_F = jnp.zeros((F_current.shape[0],), dtype=bool)
    L_new, log_q_forward_L = q_modules.spatial.sample_and_log_q(l_key, X, no_context_L, L_current, spot_graph)
    F_new, log_q_forward_F = q_modules.loading.sample_and_log_q(f_key, X, no_context_F, F_current, pathway_graph)
    log_q_reverse_L = q_modules.spatial.log_q(L_current, X, no_context_L, spot_graph)
    log_q_reverse_F = q_modules.loading.log_q(F_current, X, no_context_F, pathway_graph)
    return L_new, F_new, log_q_forward_L + log_q_forward_F, log_q_reverse_L + log_q_reverse_F


def _mh_accept(key: jax.Array, log_ratio: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Draw an exact accept/reject decision from a log Metropolis-Hastings ratio."""
    log_acceptance = jnp.minimum(0.0, log_ratio)
    accepted = jnp.log(jax.random.uniform(key)) < log_acceptance
    return accepted, jnp.exp(log_acceptance)


# Integer move-type codes returned by `mixture_step`, since a traced branch
# cannot return a Python string.
MOVE_GLOBAL_JOINT = 0
MOVE_BLOCKED_L = 1
MOVE_BLOCKED_F = 2


def mixture_step(
    key: jax.Array,
    state: ChainState,
    beta: float,
    q_modules: AmortizedProposal,
    log_pi_fn: Callable[[jax.Array, jax.Array, ChainState], jax.Array],
    spot_zone_masks: jax.Array,
    gene_zone_masks: jax.Array,
) -> tuple[ChainState, jax.Array, jax.Array, jax.Array]:
    """Advance ``(L, F)`` by one exact mixture-kernel Metropolis-Hastings step.

    With probability ``beta`` this is a global joint jump on ``(L, F)``;
    otherwise it is a single zone-blocked update, chosen uniformly between
    one of the ``L`` spot zones in ``spot_zone_masks`` (shape ``(R, S)``,
    each row ``True`` on that zone's *complement*) and one of the ``F`` gene
    zones in ``gene_zone_masks`` (shape ``(R2, G)``, same convention). Every
    component move leaves the exact posterior invariant on its own, so the
    mixture does too; ``beta`` trades basin-hopping against within-basin
    refinement and is tuned from observed acceptance rates -- it never
    affects validity. ``log_pi_fn(L, F, state)`` must be the true target,
    independent of ``q_modules``.

    Returns the possibly-updated state, whether the move was accepted, the
    acceptance probability ``alpha_MH``, and an integer move-type code
    (``MOVE_GLOBAL_JOINT`` / ``MOVE_BLOCKED_L`` / ``MOVE_BLOCKED_F``).
    """
    global_key, branch_key, l_zone_key, f_zone_key, propose_key, accept_key = jax.random.split(key, 6)
    do_global = jax.random.uniform(global_key) < beta
    do_l_block = jax.random.bernoulli(branch_key, 0.5)
    l_mask = spot_zone_masks[jax.random.randint(l_zone_key, (), 0, spot_zone_masks.shape[0])]
    f_mask = gene_zone_masks[jax.random.randint(f_zone_key, (), 0, gene_zone_masks.shape[0])]
    current_log_pi = log_pi_fn(state.L, state.F, state)

    def global_branch(_):
        """Score a global joint (L, F) jump."""
        L_new, F_new, log_q_fwd, log_q_rev = propose_global_joint(
            propose_key, state.X, state.graph, state.pathway_graph, q_modules, state.L, state.F
        )
        log_ratio = log_pi_fn(L_new, F_new, state) + log_q_rev - current_log_pi - log_q_fwd
        return L_new, F_new, log_ratio, jnp.asarray(MOVE_GLOBAL_JOINT)

    def blocked_l_branch(_):
        """Score a zone-blocked L update."""
        L_new, log_q_fwd, log_q_rev = propose_L_block(propose_key, state.X, state.L, l_mask, state.graph, q_modules.spatial)
        log_ratio = log_pi_fn(L_new, state.F, state) + log_q_rev - current_log_pi - log_q_fwd
        return L_new, state.F, log_ratio, jnp.asarray(MOVE_BLOCKED_L)

    def blocked_f_branch(_):
        """Score a zone-blocked F update."""
        F_new, log_q_fwd, log_q_rev = propose_F_block(propose_key, state.X, state.F, f_mask, state.pathway_graph, q_modules.loading)
        log_ratio = log_pi_fn(state.L, F_new, state) + log_q_rev - current_log_pi - log_q_fwd
        return state.L, F_new, log_ratio, jnp.asarray(MOVE_BLOCKED_F)

    def blocked_branch(_):
        """Dispatch to the chosen zone-blocked move."""
        return jax.lax.cond(do_l_block, blocked_l_branch, blocked_f_branch, operand=None)

    L_candidate, F_candidate, log_ratio, move_type = jax.lax.cond(do_global, global_branch, blocked_branch, operand=None)
    accepted, alpha_mh = _mh_accept(accept_key, log_ratio)
    new_state = state._replace(
        L=jnp.where(accepted, L_candidate, state.L),
        F=jnp.where(accepted, F_candidate, state.F),
    )
    return new_state, accepted, alpha_mh, move_type
