"""Longitudinal (time-series) microbiome normalization.

Port of the David et al., *Genome Biology*, 2014 scaling recipe: samples are
rows, (z)OTUs are columns, ordered in time.
"""

from __future__ import annotations

import sys
from functools import partial

import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd

__all__ = [
    "david_recipe",
    "rel_abundance",
    "calc_jsd",
    "get_jsd_weights",
    "top_otu_indices",
    "calc_scaling_factors",
    "weighted_median",
]


def david_recipe(zotus_taxa: pd.DataFrame, *, top_frac=0.9, offset=1e-5, random_state=None):
    """Normalize (z)OTU abundances for a time series (David et al., 2014).

    Steps:

    1. raw counts -> relative abundances (row-wise)
    2. pairwise Jensen-Shannon divergence between samples
    3. restrict to the fewest OTUs making up ``top_frac`` of reads
    4. estimate a per-timepoint scaling factor in log space, weighting other
       timepoints by ``(1 - JSD) ** 2``
    5. subtract the scaling factor row-wise in log space
    6. return the scaled relative abundances in linear space

    ``offset`` is added before every ``log10``. ``random_state`` seeds the
    (result-invariant) timepoint permutation in step 4.

    Returns a DataFrame with the same index/columns as ``zotus_taxa``.
    """
    zmat = zotus_taxa.values

    rel_zmat = rel_abundance(zmat)
    jsd_mat = calc_jsd(rel_zmat)
    weight_mat = get_jsd_weights(jsd_mat)
    top_idx = top_otu_indices(zmat, top_frac=top_frac)

    log_rel_sub = np.log10(rel_zmat[:, top_idx] + offset)
    scaling_factors = calc_scaling_factors(log_rel_sub, weight_mat, random_state=random_state)

    log_rel_scaled = np.log10(rel_zmat + offset) - scaling_factors[:, np.newaxis]
    rel_zmat_scaled = np.power(10, log_rel_scaled)

    return pd.DataFrame(
        rel_zmat_scaled, index=zotus_taxa.index, columns=zotus_taxa.columns
    )


def rel_abundance(zotus_taxa: np.ndarray, axis=1):
    """Relative abundances along ``axis`` (default 1 = per sample/row)."""
    if axis == 1:
        return zotus_taxa / zotus_taxa.sum(axis=axis)[:, np.newaxis]
    elif axis == 0:
        return zotus_taxa / zotus_taxa.sum(axis=axis)
    raise ValueError(f"Do not know that axis: {axis}")


def calc_jsd(zotus_taxa: np.ndarray, base=2):
    """Square Jensen-Shannon divergence matrix between samples (rows)."""
    js = partial(ssd.jensenshannon, base=base)
    return ssd.squareform(ssd.pdist(zotus_taxa, js))


def get_jsd_weights(jsd_mat: np.ndarray):
    """Column-normalized ``(1 - JSD) ** 2`` weights with a zero diagonal."""
    weight_mat = np.power(1 - jsd_mat, 2)
    weight_mat = weight_mat / weight_mat.sum(0)[:, np.newaxis]
    np.fill_diagonal(weight_mat, 0)
    return weight_mat


def top_otu_indices(zotus_taxa: np.ndarray, top_frac=0.9):
    """Indices of the fewest OTUs whose median abundance covers ``top_frac``."""
    medians = np.median(zotus_taxa, axis=0)
    frac_medians = medians / medians.sum()
    sort_medians = np.flipud(np.sort(frac_medians))
    cumsum_medians = np.cumsum(sort_medians)
    thresh_index = np.flatnonzero(cumsum_medians > top_frac)[0]
    return np.flatnonzero(frac_medians > sort_medians[thresh_index])


def calc_scaling_factors(log_otu_mat: np.ndarray, weight_mat: np.ndarray, *, random_state=None):
    """Per-timepoint log-space scaling factor vs a weighted-median reference.

    For each timepoint the reference is the weighted median across all
    timepoints (weights = that timepoint's row of ``weight_mat``); the scaling
    factor is the median of ``observed - expected``. The permutation does not
    affect the result and is kept only for parity with the original.
    """
    n_time_points = log_otu_mat.shape[0]
    order = np.random.default_rng(random_state).permutation(n_time_points)

    scaling_factors = np.zeros(n_time_points)
    for i in order:
        weights_i = weight_mat[i]
        observed_i = log_otu_mat[i]
        expected_i = np.apply_along_axis(weighted_median, 0, log_otu_mat, weights=weights_i)
        scaling_factors[i] = np.median(observed_i - expected_i)

    return scaling_factors


def weighted_median(data, weights=None):
    """Weighted median of a 1d array; unweighted median when ``weights`` is None."""
    if weights is None:
        return np.median(np.array(data).flatten())

    data, weights = np.array(data).flatten(), np.array(weights).flatten()

    if any(weights > 0):
        sorted_data, sorted_weights = map(np.array, zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(sorted_weights)

        if any(weights > midpoint):
            return (data[weights == np.max(weights)])[0]

        cumulative_weight = np.cumsum(sorted_weights)
        below_midpoint_index = np.where(cumulative_weight <= midpoint)[0][-1]

        if np.abs(cumulative_weight[below_midpoint_index] - midpoint) < sys.float_info.epsilon:
            return np.mean(sorted_data[below_midpoint_index : below_midpoint_index + 2])

        return sorted_data[below_midpoint_index + 1]
