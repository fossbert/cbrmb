"""Generalized UniFrac distances via R's ``GUniFrac`` package."""

from __future__ import annotations

import pandas as pd

from . import require_rpy2, r_package
from ._bridge import pandas_to_rpy2, rpy2_to_numpy

_ALPHAS = [0.0, 0.5, 1.0]


def calc_gunifrac(counts, tree_path, *, alpha=0.5, strip_tip_quotes=True):
    """Generalized UniFrac distance matrix.

    Parameters
    ----------
    counts : DataFrame
        Samples in rows, OTUs in columns. Column names must match the tree tip
        labels. Pass the same table the original recipe used (the notebook fed
        Rhea-normalized counts).
    tree_path : str or path-like
        Newick tree file; it is midpoint-rooted before use.
    alpha : {0.0, 0.5, 1.0}
        GUniFrac weighting parameter; ``0.5`` (``d_0.5``) is the usual choice.
    strip_tip_quotes : bool
        Remove single quotes from tip labels (common in exported trees).

    Returns
    -------
    DataFrame
        Square distance matrix indexed/columned by ``counts.index``.
    """
    if alpha not in _ALPHAS:
        raise ValueError(f"alpha must be one of {_ALPHAS}")
    if not isinstance(counts, pd.DataFrame):
        raise TypeError("counts must be a DataFrame (samples x OTUs)")

    ro, *_ = require_rpy2()
    ape = r_package("ape")
    phangorn = r_package("phangorn")
    gunifrac = r_package("GUniFrac")

    tree = ape.read_tree(str(tree_path))
    if strip_tip_quotes:
        tips = ro.r["gsub"]("'", "", tree.rx2("tip.label"), fixed=True)
        tree[tree.names.index("tip.label")] = tips
    rooted_tree = phangorn.midpoint(tree)

    unifracs = gunifrac.GUniFrac(
        pandas_to_rpy2(counts), rooted_tree, alpha=ro.FloatVector(_ALPHAS)
    )[0]
    arr = rpy2_to_numpy(unifracs)[:, :, _ALPHAS.index(alpha)]

    return pd.DataFrame(arr, index=counts.index, columns=counts.index)
