"""p-value helpers: multiple-testing correction and significance codes."""

from __future__ import annotations

from statsmodels.stats.multitest import multipletests

__all__ = ["fdr", "cut_p"]


def fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values.

    Parameters
    ----------
    pvals : array-like
        Raw p-values. Must not contain NaN (drop them first, as the
        ``*_test`` helpers do via ``dropna(subset=["pval"])``).

    Returns
    -------
    numpy.ndarray
        Adjusted p-values, same order as the input.
    """
    return multipletests(pvals, method="fdr_bh")[1]


def cut_p(p):
    """Convert a p-value to a compact significance code.

    ``< 0.001`` -> ``'***'``, ``< 0.01`` -> ``'**'``, ``< 0.05`` -> ``'*'``,
    ``< 0.1`` -> the value formatted as ``'0.07'``, otherwise ``'ns'``.
    """
    p = float(p)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return f"{p:.2f}"
    else:
        return "ns"
