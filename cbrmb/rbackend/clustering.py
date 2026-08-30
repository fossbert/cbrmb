"""Pick a cluster count / partition with R's ``NbClust``."""

from __future__ import annotations

import pandas as pd

from . import require_rpy2, r_package
from ._bridge import numpy_to_rpy2, pandas_to_rpy2, square_array

__all__ = ["best_clusters"]


def best_clusters(counts, distmat, name: str, *, method="ward.D2", index="ch"):
    """NbClust partition on a precomputed dissimilarity.

    Parameters
    ----------
    counts : DataFrame
        Samples x features table handed to NbClust as the data matrix.
    distmat : ndarray or square DataFrame
        Precomputed pairwise sample distances (e.g. generalized UniFrac).
    name : str
        Prefix for the returned labels (``f"{name}_{k}"``).
    method, index : str
        Forwarded to ``NbClust`` (default Ward.D2 linkage, Calinski-Harabasz
        index).

    Returns
    -------
    Series
        Cluster label per sample, indexed like ``counts``.
    """
    ro, *_ = require_rpy2()
    r_stats = r_package("stats")
    nbclust = r_package("NbClust")

    diss = r_stats.as_dist(numpy_to_rpy2(square_array(distmat)))
    res = nbclust.NbClust(
        pandas_to_rpy2(counts),
        diss=diss,
        distance=ro.NULL,
        method=method,
        index=index,
    )
    labels = [f"{name}_{int(k)}" for k in res.rx2("Best.partition")]
    idx = counts.index if isinstance(counts, pd.DataFrame) else None
    return pd.Series(labels, index=idx, name="Best_clusters")
