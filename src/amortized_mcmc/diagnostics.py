"""Calibration and boundary-aware acceptance diagnostics."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.stats import norm


class CoverageReport(NamedTuple):
    """Summary of empirical interval coverage and interval width."""
    coverage: jnp.ndarray
    mean_interval_width: jnp.ndarray
    n_observations: jnp.ndarray


class BoundaryAcceptance(NamedTuple):
    """Acceptance rates and sample counts for boundary/interior strata."""
    boundary_rate: jnp.ndarray
    interior_rate: jnp.ndarray
    boundary_count: jnp.ndarray
    interior_count: jnp.ndarray


def stratify_acceptance(accepted: jnp.ndarray, boundary_score: jnp.ndarray, threshold: float) -> BoundaryAcceptance:
    """Split acceptance indicators by a morphological-boundary score."""
    boundary = boundary_score >= threshold
    accepted = jnp.asarray(accepted, dtype=jnp.float32)
    boundary_count = jnp.sum(boundary)
    interior_count = jnp.sum(~boundary)
    return BoundaryAcceptance(
        jnp.sum(jnp.where(boundary, accepted, 0.0)) / jnp.maximum(boundary_count, 1),
        jnp.sum(jnp.where(~boundary, accepted, 0.0)) / jnp.maximum(interior_count, 1),
        boundary_count,
        interior_count,
    )


def empirical_coverage(
    truth: jnp.ndarray,
    mean: jnp.ndarray,
    scale: jnp.ndarray,
    level: float = 0.9,
) -> CoverageReport:
    """Measure marginal central-normal interval coverage.

    ``truth``, ``mean``, and ``scale`` share a leading observation axis. The
    returned coverage and width retain all remaining axes.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")
    z = norm.ppf((1.0 + level) / 2.0)
    lower = mean - z * scale
    upper = mean + z * scale
    covered = (truth >= lower) & (truth <= upper)
    return CoverageReport(
        jnp.mean(covered, axis=0),
        jnp.mean(upper - lower, axis=0),
        jnp.asarray(truth.shape[0]),
    )
