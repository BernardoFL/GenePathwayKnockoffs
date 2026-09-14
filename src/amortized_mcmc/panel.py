"""Stable gene-panel identity checks bridging the simulator and real data."""

from __future__ import annotations

import numpy as np


def assert_gene_panel_matches(simulated_gene_ids, real_gene_ids) -> None:
    """Assert the simulator's gene panel exactly matches a real dataset's.

    ``F``, ``A_path``, and ``gene_ids`` are all aligned along the same gene
    axis by position, not by re-looking-up an identifier, so a silent panel
    mismatch (different genes, or the same genes in a different order)
    would corrupt every ``F_g`` interpretation downstream. Call this before
    training data generated for one gene panel is used against a dataset
    with a different one.
    """
    simulated_gene_ids = np.asarray(simulated_gene_ids)
    real_gene_ids = np.asarray(real_gene_ids)
    if simulated_gene_ids.shape != real_gene_ids.shape or not np.array_equal(simulated_gene_ids, real_gene_ids):
        raise ValueError(
            "simulator gene panel does not match the real dataset's gene panel "
            "(same identifiers, same order required); regenerate training data "
            "against this dataset's post-QC gene set before training or scoring against it"
        )
