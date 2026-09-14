"""Public API for amortized neural MCMC over spatial factor models."""

from .graph import Graph, build_graph, build_pathway_graph, assert_graphs_not_aliased
from .models import AmortizedProposal, GATLayer, LoadingProposal, SpatialProposal
from .sampler import (
    MOVE_BLOCKED_F,
    MOVE_BLOCKED_L,
    MOVE_GLOBAL_JOINT,
    ChainState,
    mixture_step,
    propose_F_block,
    propose_global_joint,
    propose_L_block,
)
from .numpyro_model import numpyro_log_pi_L, spatial_model
from .target import (
    csp_log_prior_F,
    log_pi,
    log_pi_block_F,
    log_pi_block_L,
    log_pi_F,
    log_pi_F_joint,
    log_pi_L,
    poisson_log_likelihood,
    laplace_mrf_logprob,
    pathway_laplacian_from_affinity,
    pathway_laplacian_rank,
    pathway_logprob,
    pathway_quadratic_forms,
)
from .alpha import gibbs_step_alpha, gibbs_step_alpha_p, gibbs_step_sigma_alpha_sq
from .panel import assert_gene_panel_matches
from .pathway import gibbs_step_rho
from .diagnostics import empirical_coverage, stratify_acceptance
from .csp import CSPState, gibbs_step_csp, sample_csp_prior
from .flow import (
    AffineCouplingLayer,
    CouplingFlow,
    GraphCouplingLayer,
    masked_normal_logpdf,
    masked_student_t_logpdf,
    standard_normal_logpdf,
    student_t_logpdf,
)
from .mixing import compare_mixing, effective_sample_size
from .generative_model import full_generative_model
from .zoning import spectral_zone_masks

__all__ = [
    "AmortizedProposal",
    "AffineCouplingLayer",
    "CSPState",
    "CouplingFlow",
    "ChainState",
    "GATLayer",
    "Graph",
    "GraphCouplingLayer",
    "LoadingProposal",
    "SpatialProposal",
    "MOVE_BLOCKED_F",
    "MOVE_BLOCKED_L",
    "MOVE_GLOBAL_JOINT",
    "assert_gene_panel_matches",
    "assert_graphs_not_aliased",
    "build_graph",
    "build_pathway_graph",
    "empirical_coverage",
    "compare_mixing",
    "csp_log_prior_F",
    "gibbs_step_alpha",
    "gibbs_step_alpha_p",
    "gibbs_step_csp",
    "gibbs_step_rho",
    "gibbs_step_sigma_alpha_sq",
    "full_generative_model",
    "laplace_mrf_logprob",
    "log_pi_block_F",
    "log_pi_block_L",
    "log_pi_L",
    "log_pi",
    "log_pi_F",
    "log_pi_F_joint",
    "pathway_laplacian_from_affinity",
    "masked_normal_logpdf",
    "masked_student_t_logpdf",
    "mixture_step",
    "numpyro_log_pi_L",
    "poisson_log_likelihood",
    "pathway_laplacian_rank",
    "pathway_logprob",
    "pathway_quadratic_forms",
    "propose_F_block",
    "propose_global_joint",
    "propose_L_block",
    "sample_csp_prior",
    "spectral_zone_masks",
    "standard_normal_logpdf",
    "student_t_logpdf",
    "stratify_acceptance",
    "spatial_model",
    "effective_sample_size",
]
