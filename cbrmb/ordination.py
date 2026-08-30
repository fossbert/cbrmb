"""2D embeddings of a precomputed sample-distance matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

__all__ = ["calc_mds", "calc_umap"]


def _as_distance_array(dist, index):
    """Return ``(ndarray, index)``; a square DataFrame contributes its index."""
    if isinstance(dist, pd.DataFrame):
        if index is None:
            index = dist.index
        dist = dist.values
    return np.asarray(dist), index


def _embedding_frame(emb, s1, index):
    out = pd.DataFrame(emb, columns=["Dim1", "Dim2"], index=index)
    if s1 is not None:
        out["s1"] = list(s1)
    return out


def calc_mds(dist, s1=None, *, metric=False, index=None, random_state=42, **mds_kwargs):
    """Multidimensional scaling of a precomputed distance matrix.

    Parameters
    ----------
    dist : ndarray or square DataFrame
        Precomputed pairwise distances. A DataFrame's index is used to label
        the output unless ``index`` is given.
    s1 : sequence, optional
        Per-sample grouping copied into a trailing ``s1`` column.
    metric : bool
        ``False`` (default) runs non-metric MDS, as in the original helper.
    index : sequence, optional
        Row labels for the returned frame (e.g. ``adata.obs_names``).
    random_state : int
    **mds_kwargs
        Forwarded to :class:`sklearn.manifold.MDS`.

    Returns
    -------
    DataFrame with columns ``Dim1``, ``Dim2`` and optionally ``s1``.
    """
    dist, index = _as_distance_array(dist, index)
    mds = MDS(
        n_components=2,
        metric=metric,
        dissimilarity="precomputed",
        random_state=random_state,
        **mds_kwargs,
    )
    return _embedding_frame(mds.fit_transform(dist), s1, index)


def calc_umap(dist, s1=None, *, index=None, random_state=42, **umap_kwargs):
    """UMAP embedding of a precomputed distance matrix.

    Same inputs/outputs as :func:`calc_mds`. ``**umap_kwargs`` go to
    :class:`umap.UMAP` (e.g. ``min_dist=0.5``). Needs the ``umap`` extra.
    """
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "calc_umap needs the 'umap' extra: pip install 'cbrmb[umap]'"
        ) from exc

    dist, index = _as_distance_array(dist, index)
    emb = UMAP(metric="precomputed", random_state=random_state, **umap_kwargs).fit_transform(dist)
    return _embedding_frame(emb, s1, index)
