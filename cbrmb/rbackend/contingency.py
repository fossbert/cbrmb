"""r x c Fisher's exact test via R's ``stats::fisher_test``."""

from __future__ import annotations

import numpy as np

from ._bridge import numpy_to_rpy2
from . import r_package


def fisher_exact_rc(table, workspace=2e8):
    """Fisher's exact test on an arbitrary r x c contingency table.

    Parameters
    ----------
    table : 2d array-like of int
        Contingency counts.
    workspace : int
        Passed to ``fisher.test`` for larger tables.

    Returns
    -------
    (odds_ratio, p_value) : tuple of float
        ``odds_ratio`` is NaN for tables larger than 2x2 (R does not report a
        single OR there), matching :func:`scipy.stats.fisher_exact`'s 2x2 output
        shape.
    """
    r_stats = r_package("stats")
    res = r_stats.fisher_test(numpy_to_rpy2(np.asarray(table)), workspace=workspace)
    pval = res.rx2("p.value")[0]
    orr = res.rx2("estimate")
    odds_ratio = orr[0] if len(orr) else np.nan
    return float(odds_ratio), float(pval)
