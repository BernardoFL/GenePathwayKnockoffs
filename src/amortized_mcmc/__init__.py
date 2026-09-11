"""Public API for amortized neural MCMC over spatial factor models."""

from .graph import Graph, build_graph
from .models import AmortizedProposal, LoadingProposal, SpatialProposal
from .sampler import ChainState, blackjax_mh_step, mh_step, propose_F, propose_L
from .numpyro_model import numpyro_log_pi_L, spatial_model
from .target import csp_log_prior_F, log_pi_F, log_pi_F_joint, log_pi_L, poisson_log_likelihood, laplace_mrf_logprob
from .alpha import gibbs_step_alpha
from .diagnostics import empirical_coverage, stratify_acceptance
from .csp import CSPState, gibbs_step_csp, sample_csp_prior
from .flow import conditional_flow_matching_loss, divergence_hutchinson, integrate_log_density, integrate_ode
from .mixing import compare_mixing, effective_sample_size
from .generative_model import full_generative_model

__all__ = [
    "AmortizedProposal",
    "CSPState",
    "ChainState",
    "Graph",
    "LoadingProposal",
    "SpatialProposal",
    "build_graph",
    "blackjax_mh_step",
    "empirical_coverage",
    "conditional_flow_matching_loss",
    "compare_mixing",
    "csp_log_prior_F",
    "divergence_hutchinson",
    "gibbs_step_alpha",
    "gibbs_step_csp",
    "full_generative_model",
    "integrate_log_density",
    "integrate_ode",
    "laplace_mrf_logprob",
    "log_pi_L",
    "log_pi_F",
    "log_pi_F_joint",
    "mh_step",
    "numpyro_log_pi_L",
    "poisson_log_likelihood",
    "propose_F",
    "propose_L",
    "sample_csp_prior",
    "stratify_acceptance",
    "spatial_model",
    "effective_sample_size",
]
