"""Feature filtering for count / relative-abundance tables.

All functions take a wide table ``d`` with samples in rows and features in
columns, as a :class:`pandas.DataFrame` or a 2d :class:`numpy.ndarray`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "filter_prevalence",
    "filter_rel_abundance",
    "filter_features",
]


def _apply_mask(d, mask, return_mask):
    if return_mask:
        return mask
    if isinstance(d, pd.DataFrame):
        return d.loc[:, mask].copy()
    return np.asarray(d)[:, np.asarray(mask)]


def filter_prevalence(d, prevalence=0.1, *, verbose=False, return_mask=False):
    """Keep features present (value > 0) in at least ``prevalence`` of samples.

    Parameters
    ----------
    d : DataFrame or ndarray
    prevalence : float
        Minimum fraction of samples with a non-zero value.
    verbose : bool
        Print how many features were kept.
    return_mask : bool
        Return the boolean column mask instead of the filtered table.
    """
    mask = (d > 0).mean(0) >= prevalence
    if verbose:
        print(f"Kept {int(np.sum(np.asarray(mask)))} out of {d.shape[1]} features.")
    return _apply_mask(d, mask, return_mask)


def filter_rel_abundance(d, max_rel=0.25, frac=1, *, verbose=False, return_mask=False):
    """Drop features that never rise above a low-abundance floor.

    A feature is *dropped* when it is ``<= max_rel`` in at least ``frac`` of the
    samples; equivalently it is *kept* when the fraction of samples in which it
    stays ``<= max_rel`` is strictly below ``frac``. With ``frac=1`` this drops
    only features that are ``<= max_rel`` in every sample.

    Parameters
    ----------
    d : DataFrame or ndarray
        Relative-abundance table (same units as ``max_rel``).
    max_rel : float
        Low-abundance threshold.
    frac : float
        Fraction-of-samples cutoff for the drop rule.
    verbose, return_mask : bool
        See :func:`filter_prevalence`.
    """
    mask = (d <= max_rel).mean(0) < frac
    if verbose:
        print(f"Kept {int(np.sum(np.asarray(mask)))} of {d.shape[1]} features.")
    return _apply_mask(d, mask, return_mask)


def filter_features(
    d,
    *,
    prevalence=0.1,
    max_rel=0.25,
    frac=1,
    verbose=False,
    return_mask=False,
):
    """Combine the prevalence and relative-abundance filters (logical AND).

    This is the shared core behind the AnnData-aware ``filter_zotu`` /
    ``filter_tax`` adapters.
    """
    keep = filter_prevalence(d, prevalence, return_mask=True) & filter_rel_abundance(
        d, max_rel, frac, return_mask=True
    )
    if verbose:
        print(f"Kept {int(np.sum(np.asarray(keep)))} out of {d.shape[1]} features.")
    return _apply_mask(d, keep, return_mask)
