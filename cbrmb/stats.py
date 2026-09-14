"""Feature-wise group-comparison and correlation tests.

Every function takes a wide :class:`pandas.DataFrame` ``df`` (samples in rows,
features in columns) and a ``ref`` grouping or continuous vector aligned to the
rows of ``df``. Each returns a DataFrame indexed by feature with at least
``pval`` and (BH-adjusted) ``fdr`` columns, sorted by ``pval`` ascending.
Features whose test could not be evaluated (``pval`` is NaN) are dropped.
"""

from __future__ import annotations

import warnings
from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu as _mwu, pearsonr, spearmanr, wilcoxon

from .pvalues import fdr

__all__ = [
    "kruskal_test",
    "mwu_test",
    "wilcoxon_test",
    "corr_test",
    "fisher_test",
    "mwu_one_vs_rest",
    "mwu_pairwise",
]


def _finalize(df_out):
    """Drop NaN p-values, insert ``fdr`` right after ``pval``, sort by ``pval``."""
    df_out = df_out.dropna(subset=["pval"])
    adjusted = fdr(df_out["pval"]) if len(df_out) else []
    df_out.insert(df_out.columns.get_loc("pval") + 1, "fdr", adjusted)
    return df_out.sort_values("pval")


def _two_level_masks(df, ref):
    """Boolean row-masks for a ``ref`` with exactly two levels, in row order."""
    dummies = pd.get_dummies(ref).astype(bool)
    if dummies.shape[1] != 2:
        raise ValueError("Need ref with exactly two levels!")
    idx = np.arange(len(df))
    left, right = dummies.iloc[:, 0], dummies.iloc[:, 1]
    return df.iloc[idx[left.values]], df.iloc[idx[right.values]]


def kruskal_test(df: pd.DataFrame, ref, quantile=0.75):
    """Kruskal-Wallis test per feature across the levels of ``ref`` (>= 2).

    Returns a DataFrame with ``pval``, ``fdr``, ``stat`` and one column per
    level of ``ref`` holding that level's ``quantile`` of the feature.
    """
    out = []
    for _, v in df.items():
        vg = v.groupby(ref, observed=True)
        stat, pval = kruskal(*vg.apply(lambda x: x.dropna().values))
        qs = vg.quantile(q=quantile)
        out.append((pval, stat, *qs))

    df_out = pd.DataFrame(
        out, index=df.columns, columns=["pval", "stat", *qs.index.to_list()]
    )
    return _finalize(df_out)


def mwu_test(df: pd.DataFrame, ref, quantile=0.75):
    """Mann-Whitney-U test per feature between the two levels of ``ref``.

    Returns ``pval``, ``fdr``, ``u1``, ``u2`` and one ``quantile`` column per
    level. NaNs are handled per feature (``nan_policy="omit"``).
    """
    arr1, arr2 = _two_level_masks(df, ref)

    u1, pvals = _mwu(arr1, arr2, axis=0, nan_policy="omit")
    u2 = arr1.shape[0] * arr2.shape[0] - u1

    qs = df.groupby(ref, observed=True).quantile(q=quantile).T

    df_out = pd.DataFrame(
        np.column_stack([pvals, u1, u2, qs.values]),
        index=df.columns,
        columns=["pval", "u1", "u2", *qs.columns.to_list()],
    )
    return _finalize(df_out)


def wilcoxon_test(df: pd.DataFrame, ref, quantile=0.75):
    """Wilcoxon signed-rank test per feature between two *paired* levels of ``ref``.

    The two groups are matched by row order within each group, so the caller is
    responsible for arranging ``df`` so that row *i* of the first level pairs
    with row *i* of the second level.

    Returns ``stat``, ``pval``, ``fdr`` and one ``quantile`` column per level.
    """
    warnings.warn(
        "wilcoxon_test assumes paired data matched by within-group row order; "
        "make sure df is arranged accordingly.",
        stacklevel=2,
    )
    arr1, arr2 = _two_level_masks(df, ref)

    res = wilcoxon(arr1.values, arr2.values, axis=0, nan_policy="omit")
    stat, pvals = np.atleast_1d(res.statistic), np.atleast_1d(res.pvalue)

    qs = df.groupby(ref, observed=True).quantile(q=quantile).T

    df_out = pd.DataFrame(
        np.column_stack([stat, pvals, qs.values]),
        index=df.columns,
        columns=["stat", "pval", *qs.columns.to_list()],
    )
    return _finalize(df_out)


def corr_test(df: pd.DataFrame, ref, method="spearman"):
    """Correlate every feature in ``df`` with the continuous vector ``ref``.

    ``method`` is ``"spearman"`` (default) or ``"pearson"``. Returns ``coef``,
    ``fdr``, ``pval``.
    """
    method_options = {
        "pearson": pearsonr,
        "spearman": partial(spearmanr, nan_policy="omit"),
    }
    if method not in method_options:
        raise ValueError(f"method must be one of {sorted(method_options)}")
    corr_method = method_options[method]

    out = [corr_method(ref, v) for _, v in df.items()]

    df_out = pd.DataFrame(out, index=df.columns, columns=["coef", "pval"])
    return _finalize(df_out)


def fisher_test(df: pd.DataFrame, ref):
    """Fisher's exact test per feature between ``ref`` and each column of ``df``.

    Each column of ``df`` is cross-tabulated against ``ref``. 2x2 tables use
    :func:`scipy.stats.fisher_exact`; larger tables fall back to R's
    ``stats::fisher_test`` (requires the ``r`` extra). Features whose table has
    a dimension < 2 yield NaN and are dropped. Returns ``pval``, ``fdr``,
    ``odds_ratio`` (NaN for non-2x2 tables from the R path).
    """
    from scipy.stats import fisher_exact

    _rc_fisher = None
    out = []
    for _, v in df.items():
        tab = pd.crosstab(ref, v)
        # pd.crosstab has no ``observed`` argument; emulate it by dropping
        # empty rows/columns left behind by unused categorical levels.
        tab = tab.loc[(tab != 0).any(axis=1), (tab != 0).any(axis=0)]
        if not all(dim >= 2 for dim in tab.shape):
            out.append((np.nan, np.nan))
            continue
        if tab.shape == (2, 2):
            orr, pval = fisher_exact(tab.values)
        else:
            if _rc_fisher is None:
                from .rbackend.contingency import fisher_exact_rc as _rc_fisher
            orr, pval = _rc_fisher(tab.values)
        out.append((pval, orr))

    df_out = pd.DataFrame(out, index=df.columns, columns=["pval", "odds_ratio"])
    return _finalize(df_out)


def mwu_one_vs_rest(df: pd.DataFrame, ref, adjust_p=True, verbose=True, **mwu_kwargs):
    """One-vs-rest Mann-Whitney-U for every level of ``ref``.

    Returns ``(df_pval, df_diff)``: p-values (BH-adjusted per level when
    ``adjust_p``) and the difference in column means (level minus rest), both
    indexed by feature with ``"<level>_vs_rest"`` columns.
    """
    dummies = pd.get_dummies(ref).astype(bool)
    idx = np.arange(len(dummies))

    res_pval, res_diff = {}, {}
    for k, v in dummies.items():
        arr1 = df.iloc[idx[v.values]]
        arr2 = df.iloc[idx[~v.values]]

        pvals = _mwu(arr1, arr2, axis=0, **mwu_kwargs)[1]
        diff_means = np.nanmean(arr1, axis=0) - np.nanmean(arr2, axis=0)

        if adjust_p:
            if verbose:
                print(f"Adjusting p-values for {k}!")
            pvals = fdr(pvals)

        res_pval[k] = pvals
        res_diff[k] = diff_means

    df_pval = pd.DataFrame(res_pval, index=df.columns).add_suffix("_vs_rest")
    df_diff = pd.DataFrame(res_diff, index=df.columns).add_suffix("_vs_rest")
    return df_pval, df_diff


def mwu_pairwise(df: pd.DataFrame, ref, adjust_p=True, verbose=True, **mwu_kwargs):
    """All pairwise Mann-Whitney-U comparisons between the levels of ``ref``.

    Returns ``(df_pvals, df_diff)`` with one ``"<a>_vs_<b>"`` column per pair;
    p-values are BH-adjusted per pair when ``adjust_p``. ``df_diff`` holds the
    difference in column means (a minus b).
    """
    dummies = pd.get_dummies(ref).astype(bool)
    idx = np.arange(len(dummies))

    res_pvals, res_diff = {}, {}
    for c1, c2 in combinations(dummies.columns.to_list(), 2):
        pair_id = f"{c1}_vs_{c2}"

        arr1 = df.iloc[idx[dummies[c1].values]]
        arr2 = df.iloc[idx[dummies[c2].values]]

        pvals = _mwu(arr1, arr2, axis=0, **mwu_kwargs)[1]
        res_diff[pair_id] = np.nanmean(arr1, axis=0) - np.nanmean(arr2, axis=0)

        if adjust_p:
            if verbose:
                print(f"Adjusting p-values for {pair_id}!")
            pvals = fdr(pvals)

        res_pvals[pair_id] = pvals

    df_pvals = pd.DataFrame(res_pvals, index=df.columns)
    df_diff = pd.DataFrame(res_diff, index=df.columns)
    return df_pvals, df_diff
