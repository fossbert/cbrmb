"""Distance-based confounder screening via PERMANOVA (``GUniFrac::adonis3``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import require_rpy2, r_package
from ._bridge import numpy_to_rpy2, pandas_to_rpy2, rpy2_to_pandas, square_array

__all__ = ["screen_confounder", "test_confounder", "remove_confounder_nan"]


def screen_confounder(distmat, confounders: pd.DataFrame, seed: int = 42, *, verbose=True):
    """Run :func:`test_confounder` for every column of ``confounders``.

    Columns with a single level (after dropping NaN) are skipped. Returns a
    DataFrame with columns ``var``, ``r2``, ``pval``.
    """
    rows = []
    for name, values in confounders.items():
        if values.nunique() <= 1:
            if verbose:
                print(f"Dropping {name}, levels not suitable")
            continue
        rows.append((name, *test_confounder(distmat, values, seed)))
    return pd.DataFrame(rows, columns=("var", "r2", "pval"))


def test_confounder(distmat, confounder: pd.Series, seed: int = 42):
    """PERMANOVA of one covariate against a precomputed distance matrix.

    ``distmat`` may be an ndarray or a square DataFrame. NaNs in ``confounder``
    (and the matching rows/cols of ``distmat``) are dropped first. Returns the
    ``R2`` and ``Pr(>F)`` entries of the adonis term row.
    """
    ro, *_ = require_rpy2()
    from rpy2.robjects import Formula

    r_stats = r_package("stats")
    gunifrac = r_package("GUniFrac")

    distmat = square_array(distmat)
    if confounder.isnull().any():
        distmat, confounder = remove_confounder_nan(distmat, confounder)

    fmla = Formula("y ~ x")
    env = fmla.environment
    env["y"] = r_stats.as_dist(numpy_to_rpy2(distmat))
    env["x"] = pandas_to_rpy2(confounder)

    ro.r("set.seed")(seed)
    res = rpy2_to_pandas(gunifrac.adonis3(formula=fmla)[0])
    return res.iloc[0, 4:]


def remove_confounder_nan(distmat, confounder: pd.Series):
    """Drop NaN rows of ``confounder`` and the matching rows/cols of ``distmat``."""
    distmat = square_array(distmat)
    idx = np.arange(distmat.shape[0])[confounder.notnull().values]
    return distmat[idx[:, np.newaxis], idx], confounder.iloc[idx]
